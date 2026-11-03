"""
Sudoku Validator
================
A single-file Streamlit app that checks whether a completed 9x9 Sudoku board
is valid, using NumPy for all of the grid logic.

Run locally:   streamlit run app.py
"""

import io

import numpy as np
import pandas as pd  # installed automatically as a dependency of Streamlit
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SIZE = 9            # The board is SIZE x SIZE
BOX = 3             # Each sub-grid is BOX x BOX
TARGET_SUM = 45     # 1 + 2 + ... + 9 = 45
DIGITS = np.arange(1, 10)  # array([1, 2, ..., 9]), used for the uniqueness check


# --------------------------------------------------------------------------
# Core logic (pure NumPy, no Streamlit calls -> easy to unit test)
# --------------------------------------------------------------------------
def to_board(data) -> tuple[np.ndarray | None, list[str]]:
    """
    Convert raw input (DataFrame / list / ndarray) into a clean int 9x9 board.

    Returns (board, errors). If errors is non-empty, board is None.
    Checks performed:
      * shape is exactly 9x9
      * every cell is a real number (no blanks / text / NaN)
      * every number is a whole number (no 3.5)
      * every number is between 1 and 9 (so 0 = "empty cell" is rejected)
    """
    errors: list[str] = []

    # Coerce everything to float first so blanks/text become NaN instead of
    # crashing. errors="coerce" turns unparseable values into NaN.
    try:
        arr = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        return None, [f"Could not read the board: {exc}"]

    # --- Shape check ------------------------------------------------------
    if arr.shape != (SIZE, SIZE):
        return None, [f"Board must be exactly 9x9, but got {arr.shape[0]}x{arr.shape[1]}."]

    # --- Non-numeric / blank cells ---------------------------------------
    # np.isnan returns a boolean matrix; np.argwhere lists (row, col) of True.
    for r, c in np.argwhere(np.isnan(arr)):
        errors.append(f"Row {r + 1}, Column {c + 1}: cell is blank or not a number.")
    if errors:
        return None, errors

    # --- Whole numbers only ----------------------------------------------
    # arr != np.floor(arr) is True wherever a value has a fractional part.
    for r, c in np.argwhere(arr != np.floor(arr)):
        errors.append(f"Row {r + 1}, Column {c + 1}: {arr[r, c]} is not a whole number.")

    # --- Range 1..9 -------------------------------------------------------
    # Boolean mask: True wherever a value is outside [1, 9]. Zero (an empty
    # cell) is caught here too.
    for r, c in np.argwhere((arr < 1) | (arr > 9)):
        errors.append(
            f"Row {r + 1}, Column {c + 1}: value {arr[r, c]:g} is outside 1-9."
        )
    if errors:
        return None, errors

    return arr.astype(int), []


def _is_complete_set(values: np.ndarray) -> bool:
    """
    True if `values` contains each of 1..9 exactly once.

    Why not just check sum == 45? Because the sum test alone is NOT sufficient:
    [1,3,4,5,6,5,6,7,8] also adds up to 45 but contains duplicates. We keep the
    required sum checks below, and use this as a second line of defence.
    """
    return np.array_equal(np.sort(values.ravel()), DIGITS)


def check_rows(board: np.ndarray) -> list[str]:
    """Validate every row. Returns a list of human-readable failures."""
    failures = []

    # np.sum(board, axis=1) collapses the COLUMNS and returns one total per ROW:
    #   board = [[a b c ...],      -> row_sums = [a+b+c+..., ...]
    #            [d e f ...]]
    row_sums = np.sum(board, axis=1)

    for r in range(SIZE):
        if row_sums[r] != TARGET_SUM:
            failures.append(f"Row {r + 1} sums to {row_sums[r]} instead of {TARGET_SUM}.")
        elif not _is_complete_set(board[r, :]):  # board[r, :] = the whole r-th row
            failures.append(f"Row {r + 1} sums to 45 but contains duplicate digits.")
    return failures


def check_columns(board: np.ndarray) -> list[str]:
    """Validate every column. Returns a list of human-readable failures."""
    failures = []

    # np.sum(board, axis=0) collapses the ROWS and returns one total per COLUMN.
    col_sums = np.sum(board, axis=0)

    for c in range(SIZE):
        if col_sums[c] != TARGET_SUM:
            failures.append(f"Column {c + 1} sums to {col_sums[c]} instead of {TARGET_SUM}.")
        elif not _is_complete_set(board[:, c]):  # board[:, c] = the whole c-th column
            failures.append(f"Column {c + 1} sums to 45 but contains duplicate digits.")
    return failures


def check_subgrids(board: np.ndarray) -> list[str]:
    """
    Validate the nine 3x3 sub-grids.

    Slicing refresher:  board[i:i+3, j:j+3]
      * i:i+3  -> take rows    i, i+1, i+2
      * j:j+3  -> take columns j, j+1, j+2
    Together that selects a 3x3 "window" whose top-left corner is (i, j).

    Looping i and j with a step of 3 visits the corners 0, 3, 6 in each
    direction, so the windows tile the board with no overlap:

        (0,0) (0,3) (0,6)
        (3,0) (3,3) (3,6)
        (6,0) (6,3) (6,6)
    """
    failures = []
    for i in range(0, SIZE, BOX):          # i = 0, 3, 6  (grid row)
        for j in range(0, SIZE, BOX):      # j = 0, 3, 6  (grid column)
            block = board[i:i + BOX, j:j + BOX]   # the 3x3 window
            # Human-friendly name: grid row/col numbered 1..3
            name = f"Sub-grid (row {i // BOX + 1}, column {j // BOX + 1})"

            if block.sum() != TARGET_SUM:
                failures.append(f"{name} sums to {block.sum()} instead of {TARGET_SUM}.")
            elif not _is_complete_set(block):
                failures.append(f"{name} sums to 45 but contains duplicate digits.")
    return failures


def validate_board(board: np.ndarray) -> list[str]:
    """Run all three checks and return every failure found (empty = valid)."""
    return check_rows(board) + check_columns(board) + check_subgrids(board)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def read_csv_board(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded CSV into a DataFrame.
    header=None means "the first line is data, not column names", which is
    what you want for a bare 9x9 grid of digits. If the CSV has a header row,
    the text cells will be flagged as errors by to_board().
    """
    text = io.StringIO(uploaded_file.getvalue().decode("utf-8-sig"))
    return pd.read_csv(text, header=None)


def main() -> None:
    st.set_page_config(page_title="Sudoku Validator", page_icon="🧩", layout="centered")
    st.title("🧩 Sudoku Validator")
    st.write(
        "Enter a completed 9x9 Sudoku (or upload a CSV) and check whether every "
        "row, column and 3x3 sub-grid is valid."
    )

    mode = st.radio("Input method", ["Manual entry", "CSV upload"], horizontal=True)

    raw_board = None  # whatever the user gave us; validated on button click

    if mode == "Manual entry":
        st.caption("Click a cell and type a digit 1-9. Zeros are treated as empty cells.")
        # Start with a 9x9 grid of zeros. Column names are 1..9 for readability.
        start = pd.DataFrame(
            np.zeros((SIZE, SIZE), dtype=int),
            columns=[str(c) for c in range(1, SIZE + 1)],
        )
        raw_board = st.data_editor(
            start,
            key="grid",
            hide_index=False,  # row numbers on the left
            use_container_width=True,
            column_config={
                str(c): st.column_config.NumberColumn(min_value=0, max_value=9, step=1)
                for c in range(1, SIZE + 1)
            },
        )
    else:
        st.caption("Upload a CSV with 9 rows and 9 comma-separated digits, no header.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                raw_board = read_csv_board(uploaded)
                st.write("Preview:")
                st.dataframe(raw_board, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not parse that CSV: {exc}")

    if st.button("Validate Board", type="primary"):
        if raw_board is None:
            st.warning("Please provide a board first.")
            return

        # Step 1: shape + type + range validation
        board, input_errors = to_board(raw_board)
        if input_errors:
            st.error("Invalid input:\n\n" + "\n".join(f"- {e}" for e in input_errors))
            return

        # Step 2: Sudoku rules
        failures = validate_board(board)
        if failures:
            st.error("❌ Invalid Sudoku:\n\n" + "\n".join(f"- {f}" for f in failures))
        else:
            st.success("✅ Valid Sudoku! All rows, columns and 3x3 sub-grids check out.")


if __name__ == "__main__":
    main()
