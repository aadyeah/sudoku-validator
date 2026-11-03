# sudoku-validator
A Streamlit web app that uses NumPy matrix slicing to instantly check whether a 9x9 Sudoku board is valid, and tells you exactly which row, column, or sub-grid fails.

## Features
- Manual entry via an editable 9x9 grid (`st.data_editor`)
- CSV upload (9 rows x 9 comma-separated digits, no header)
- Input validation: exact 9x9 shape, whole numbers only, values 1-9
- Row, column, and 3x3 sub-grid checks using NumPy sums and slicing
- Precise error messages (e.g. "Row 3 sums to 44 instead of 45")

## How it works
| Check | NumPy |
|---|---|
| Rows | `np.sum(board, axis=1) == 45` |
| Columns | `np.sum(board, axis=0) == 45` |
| Sub-grids | `board[i:i+3, j:j+3].sum() == 45` in nested loops with step 3 |

A sum of 45 is necessary but not sufficient (`[1,3,4,5,6,5,6,7,8]` also sums to 45), so each unit is additionally checked for the digits 1-9 appearing exactly once.

## Run locally
```bash
git clone https://github.com/<your-username>/sudoku-validator.git
cd sudoku-validator
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a **public** GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** and choose **Deploy a public app from GitHub**.
4. Select your repository, the `main` branch, and set the main file path to `app.py`.
5. Click **Deploy**. Your app gets a public `*.streamlit.app` URL; add it to this README.

## Sample CSV
A valid board you can save as `valid.csv` to test uploads:
```
5,3,4,6,7,8,9,1,2
6,7,2,1,9,5,3,4,8
1,9,8,3,4,2,5,6,7
8,5,9,7,6,1,4,2,3
4,2,6,8,5,3,7,9,1
7,1,3,9,2,4,8,5,6
9,6,1,5,3,7,2,8,4
2,8,7,4,1,9,6,3,5
3,4,5,2,8,6,1,7,9
```

## Tech stack
Python, NumPy, Streamlit
