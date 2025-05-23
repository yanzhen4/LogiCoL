import torch
from utils.tensor_utils import logsumexp, small_val
import torch.nn.functional as F

def normalize_similarity(similarity_matrix):
    """
    Normalize a similarity matrix such that the most similar element per row is 1
    and the least similar element per row is 0 using min-max normalization.
    """
    min_values = similarity_matrix.min(dim=1, keepdim=True)[0]
    max_values = similarity_matrix.max(dim=1, keepdim=True)[0]
    normalized_matrix = (similarity_matrix - min_values) / (max_values - min_values + 1e-8)  # Avoid division by zero
    return normalized_matrix

def normalize_similarity_2norm(similarity_matrix):
    """
    Normalize a similarity matrix such that the most similar element per row is 1
    and the least similar element per row is 0 using 2-norm normalization.
    """
    # print("Using 2-norm normalization")
    norm = similarity_matrix.norm(dim=1, keepdim=True)
    norm = norm.clamp(min=1e-8)
    return similarity_matrix / norm

def create_supercon_pos_neg_masks(pos_query_indices):    
    batch_size = len(pos_query_indices)

    pos_mask = torch.zeros((batch_size, batch_size), dtype=int)

    query_to_documents = {}
    for doc_id, queries in enumerate(pos_query_indices):
        for query in queries:
            if query not in query_to_documents:
                query_to_documents[query] = []
            query_to_documents[query].append(doc_id)

    for doc_list in query_to_documents.values():
        for i in range(len(doc_list)):
            for j in range(i, len(doc_list)):
                pos_mask[doc_list[i]][doc_list[j]] = 1
                pos_mask[doc_list[j]][doc_list[i]] = 1
    
    neg_mask = 1 - pos_mask

    return pos_mask, neg_mask

def create_dpr_pos_neg_masks(pos_query_indices):
    batch_size = len(pos_query_indices)
    
    pos_mask = torch.zeros((batch_size * 2, batch_size * 2), dtype=torch.int)
    neg_mask = torch.zeros((batch_size * 2, batch_size * 2), dtype=torch.int)
    
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = 1
    
    for i in range(batch_size):
        for j in range(batch_size):
            if j != i:
                neg_mask[i, j + batch_size] = 1
    
    return pos_mask, neg_mask

def supercon_loss(doc_embeds, pos_mask, neg_mask, temperature = 0.1):
    """
    SuperCon Loss 
    https://arxiv.org/abs/2004.11362

    Args:
        doc_embeds: [batch, embed_dim], document embeddings
        pos_mask: [batch, batch], mask indicating positive pairs
        temperature: float, scaling factor for the log probability smoothing
    """

    norm_embeds = doc_embeds / doc_embeds.norm(dim=1, keepdim=True)

    mat = torch.mm(norm_embeds, norm_embeds.t())

    mat = mat / temperature
    mat_max, _ = mat.max(dim=1, keepdim=True)

    mat = mat - mat_max.detach()

    denominator = logsumexp(
        mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
    )

    log_prob = mat - denominator

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
        pos_mask.sum(dim=1) + small_val(mat.dtype)
    )

    valid_mean_log_prob_pos = -mean_log_prob_pos

    #To handle dpr loss case
    # Create a mask to filter out rows where there is no positive pair
    # valid_rows_mask = pos_mask.sum(dim=1) > 0
    # valid_mean_log_prob_pos = mean_log_prob_pos[valid_rows_mask]

    # Apply the mask to mean_log_prob_pos

    low_pass_filter = valid_mean_log_prob_pos > 0
    valid_mean_log_prob_pos = valid_mean_log_prob_pos[low_pass_filter]

    loss = valid_mean_log_prob_pos.mean()
    
    return loss

def exclusion_loss(embeddings, query_indices, document_indices, exclusion_relation_mask, scale=0.5, normalization=0, margin=0.2):
    """
    Exclusion loss: penalizes exclusion-related queries that attend to similar documents.
    Encourages high KL divergence between their document similarity distributions.
    """

    query_embeddings = embeddings[query_indices]
    document_embeddings = embeddings[document_indices]

    query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
    document_embeddings = document_embeddings / document_embeddings.norm(dim=1, keepdim=True)

    q1_indices, q2_indices = torch.where(exclusion_relation_mask == 1)
    q1_embeddings = query_embeddings[q1_indices]
    q2_embeddings = query_embeddings[q2_indices]

    sim_q1_d = torch.mm(q1_embeddings, document_embeddings.T)
    sim_q2_d = torch.mm(q2_embeddings, document_embeddings.T)

    if normalization == 1:
        sim_q1_d = normalize_similarity(sim_q1_d)  
        sim_q2_d = normalize_similarity(sim_q2_d)
    else:
        sim_q1_d = torch.softmax(sim_q1_d, dim=1)
        sim_q2_d = torch.softmax(sim_q2_d, dim=1)

    sim_q1_d = torch.clamp(sim_q1_d, min=1e-8)
    sim_q2_d = torch.clamp(sim_q2_d, min=1e-8)

    kl_q1_q2 = (sim_q1_d * (torch.log(sim_q1_d) - torch.log(sim_q2_d))).sum(dim=1)
    kl_q2_q1 = (sim_q2_d * (torch.log(sim_q2_d) - torch.log(sim_q1_d))).sum(dim=1)
    sym_kl = 0.5 * (kl_q1_q2 + kl_q2_q1)
    
    constraints = torch.relu(margin - sym_kl)

    non_zero_mask_count = exclusion_relation_mask.sum().item()
    if non_zero_mask_count == 0:
        return torch.tensor(0.0, device=embeddings.device)

    return constraints.sum() * scale / non_zero_mask_count

def subset_loss_asymmetric(embeddings, query_indices, document_indices, subset_relation_mask, scale = 0.5, normalization = 0, margin = 0):
    """
    Subset loss
    """
    
    query_embeddings = embeddings[query_indices]
    document_embeddings = embeddings[document_indices]

    query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
    document_embeddings = document_embeddings / document_embeddings.norm(dim=1, keepdim=True)

    q1_indices, q2_indices = torch.where(subset_relation_mask == 1)

    q1_embeddings = query_embeddings[q1_indices]
    q2_embeddings = query_embeddings[q2_indices]

    sim_q1_d = torch.mm(q1_embeddings, document_embeddings.T)
    sim_q2_d = torch.mm(q2_embeddings, document_embeddings.T)

    if normalization == 1:
        sim_q1_d = normalize_similarity(sim_q1_d)
        sim_q2_d = normalize_similarity(sim_q2_d)
    else:
        sim_q1_d = (sim_q1_d + 1) / 2
        sim_q2_d = (sim_q2_d + 1) / 2

    sim_q1_d = torch.clamp(sim_q1_d, min=1e-8)
    sim_q2_d = torch.clamp(sim_q2_d, min=1e-8)

    constraints = torch.relu(torch.log(sim_q1_d) - torch.log(sim_q2_d) + margin)

    loss = constraints.mean(dim=1)

    non_zero_mask_count = subset_relation_mask.sum().item()

    if non_zero_mask_count == 0:
        return torch.tensor(0.0, device=embeddings.device)
        
    return loss.sum() * scale / non_zero_mask_count


def dpr_loss(q_embeds, doc_embeds, positive_idx_per_question):
    """
    Contrastive loss as used in the DPR model.
    https://arxiv.org/abs/2004.04906

    Args:
        q_embeds: [batch, embed_dim], queries encoded
        doc_embeds: [batch, embed_dim], docs encoded 
        positive_idx_per_question: [batch], index of the positive document for each query
    """
    
    # Add normalization
    q_embeds = q_embeds / q_embeds.norm(dim=1, keepdim=True)
    doc_embeds = doc_embeds / doc_embeds.norm(dim=1, keepdim=True)

    doc_embeds_t = torch.transpose(doc_embeds, 0, 1)
    scores =  torch.matmul(q_embeds, doc_embeds_t)

    if len(q_embeds.size()) > 1:
        q_num = q_embeds.size(0)
        scores = scores.view(q_num, -1)
    
    softmax_scores = F.log_softmax(scores, dim=1)

    loss = F.nll_loss(
        softmax_scores,
        positive_idx_per_question.to(softmax_scores.device),
        reduction="mean",
    )

    return loss

#TODO: Write a function that utilize the set constraints 

LOSS_CLASSES = {
    "supcon": supercon_loss,
    "dpr": dpr_loss,
    #"subset": subset_loss,
    "subset_asymmetric": subset_loss_asymmetric,
    "exclusion": exclusion_loss
}

LOSS_CLASS_PREPROCESS = {
    "supercon_loss": create_supercon_pos_neg_masks,
    "dpr_loss": create_dpr_pos_neg_masks
}
