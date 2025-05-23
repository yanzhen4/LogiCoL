import torch
from typing import Dict, List, Any, Tuple, Callable

def neg_inf(dtype):
    return torch.finfo(dtype).min

def small_val(dtype):
    return torch.finfo(dtype).tiny

def torch_arange_from_size(input, size_dim=0):
    return torch.arange(input.size(size_dim), device=input.device)

def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, neg_inf(x.dtype))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)

    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output

def reduce_embeddings(
    tensors_by_process: List[torch.Tensor],
    **kwargs
) -> torch.Tensor:
    """
    Reduce embeddings from different processes into a single tensor.
    """
    return torch.cat(tensors_by_process, dim=0)

def reduce_pos_neg_masks(
    tensors_by_process: List[torch.Tensor],
    shape_by_process: torch.Tensor,
    pad_value: int = 0,
) -> torch.Tensor:

    """
    Reduce positive and negative masks from different processes into a single tensor.
    Note that the function assumes all masks to be 2-d.
    """
    device = tensors_by_process[0].device
    total_shape = torch.sum(shape_by_process, dim=0)

    new_mask = torch.full(
        total_shape.tolist(),
        fill_value=pad_value,
        device=device)

    cum_row = 0
    cum_col = 0
    for t in tensors_by_process:
        row_size, col_size = t.shape[0], t.shape[1]

        new_mask[cum_row: cum_row + row_size, cum_col: cum_col + col_size] = t
        cum_row += row_size
        cum_col += col_size

    return new_mask

def apply_label_offset(
    tensors_by_process,
    **kwargs
) -> torch.Tensor:
    """
    Add offset to query and indicies 
    """

    offset = 0
    num_documents = 1 # This code will cause problem if we take multiple positive documents 

    for i, tensor in enumerate(tensors_by_process):
        tensors_by_process[i] = tensor + offset
        offset += len(tensor) * (num_documents + 1)

    result = torch.cat(tensors_by_process)

    return result

#TODO: Need to add offset to subset and exclusive mask 

def gather_with_var_len(tensor,
                        base_gather_fn,
                        pad_value: int = 0,
                        reduce_fn: Callable = reduce_embeddings):
    """
    Helper for gathering tensors (up to 2d) with variable size on dim=0 from multiple GPUs/accelerator nodes.

    :param tensor: name of the (sharded) tensor that we need to gather from other accelerator nodes
    :param base_gather_fn: Base gather function from pytorch/pylightning, etc
    :param pad_value: padding value for the tensor
    :param reduce_fn: function to reduce the gathered tensors
    :return:
    """
    shape = torch.tensor(tensor.shape, device=tensor.device)
    assert len(shape) == 1 or len(shape) == 2, "Only support 1d or 2d tensors for now."
    all_shape = base_gather_fn(shape)  # [num_gpu, ...shape]
    max_shape = torch.max(all_shape, dim=0).values

    if not torch.equal(shape, max_shape):
        temp_tensor = torch.full(
            max_shape.tolist(),
            fill_value=pad_value,
            dtype=tensor.dtype, 
            device=tensor.device
        )

        if len(shape) == 1:
            temp_tensor[:shape[0]] = tensor
        elif len(shape) == 2:
            temp_tensor[:shape[0], :shape[1]] = tensor
        
        tensor = temp_tensor
    
    # all gather across all processes
    all_tensors = base_gather_fn(tensor)
    # With DDP, each GPU gets its own mini-batch.
    # Since the labels are created by enumerating instances within batch, i.e. labels ranges from 0 to batch_size
    # within each batch, when merging the batches, we need to differentiate between the labels of each batch
    # This is achieved by adding a different offsets to each row of the tensor

    if len(shape) == 1:
        tensors_by_process = [all_tensors[i, 0:l, ...] for i, l in enumerate(all_shape)]
    elif len(shape) == 2:
        tensors_by_process = [all_tensors[i, 0:s[0], 0:s[1], ...] for i, s in enumerate(all_shape)]

    return reduce_fn(
        tensors_by_process=tensors_by_process, 
        shape_by_process=all_shape,
        pad_value=pad_value
    )

def make_neg_mask(pos_mask: torch.Tensor) -> torch.Tensor:
    """
    Make a negative mask from a positive mask.
    Removes the diagonal elements from the positive mask and sets the diagonal elements to 0 in the negative mask.
    """
    neg_mask =  1 - pos_mask

    neg_mask = neg_mask.fill_diagonal_(0)
    return neg_mask

def make_strict_neg_mask(pos_mask: torch.Tensor, query_indices: torch.Tensor) -> torch.Tensor:
    """
    Make a negative mask from a positive mask.
    Only the other positive documents are negative examples of a query, but not other queries 
    """
    neg_mask =  1 - pos_mask

    neg_mask = neg_mask.fill_diagonal_(0)

    # This ensures that the neg_mask for all queries are 0
    for col_idx in query_indices:
        neg_mask[:, col_idx] = 0

    return neg_mask

def sparse_indices_to_dense_matrix(
    indices: List[Tuple[int, int]], 
    shape: Tuple[int, int],
    reverse: bool = False
    ) -> torch.Tensor:
    """
    Convert a list of (row, col) indices into a dense 0/1 matrix. 
    All indices specified by (row, col) will have value of 1, and 0 otherwise.

    Args:
    - indices: List of (row, col) indices.
    - shape: Shape of the 2-D dense matrix.
    """
    mat = torch.zeros(shape)

    if len(indices) > 0:
        rows = [idx[0] for idx in indices]
        cols = [idx[1] for idx in indices]
        mat[rows, cols] = 1
    
    # If reverse is true, turn indices of 1 into 0, and 0 into 1
    if reverse:
        mat = 1 - mat

    return mat

def make_square_matrix_symmetric(mat: torch.Tensor):
    """
    Make a square matrix symmetric by taking the element-wise maximum of the matrix and its transpose.
    
    Args:
        mat: [N, N] square matrix
    """
    assert mat.shape[0] == mat.shape[1], "Input matrix must be square."
    mat_t = torch.transpose(mat, 0, 1)
    sym_mat = torch.maximum(mat, mat_t)

    return sym_mat


if __name__ == "__main__":
    indices = [(0, 1), (2, 3), (4, 5)]

    mat = sparse_indices_to_dense_matrix(indices, (6, 6))
    print(mat)

    mat = make_square_matrix_symmetric(mat)

    neg_mat = make_neg_mask(mat)

    print(mat)
    print(neg_mat)