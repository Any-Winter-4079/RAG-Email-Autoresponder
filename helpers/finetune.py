def get_condensed_index(row_index, column_index, n_groups):
    if row_index > column_index:
        row_index, column_index = column_index, row_index
    return (
        row_index * n_groups
        - row_index * (row_index + 1) // 2
        + column_index
        - row_index
        - 1
    )
