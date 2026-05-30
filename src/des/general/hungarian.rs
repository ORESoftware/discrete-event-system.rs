//! Port of `src/des/general/hungarian.ts` — Hungarian algorithm for square /
//! rectangular bipartite assignment.
//!
//! Solves   max  Σ x_{i,j} · w_{i,j}        (or min for cost matrices)
//!          s.t. Σ_j x_{i,j} = 1   ∀ i
//!               Σ_i x_{i,j} = 1   ∀ j
//!               x_{i,j} ∈ {0, 1}
//!
//! in O(n^3) where n = max(rows, cols), via the Jonker–Volgenant style
//! "shortest augmenting path" matrix variant with column potentials.
//!
//! TS used `number[][]` cost matrices and a `-1` sentinel for unmatched
//! rows/cols; here we mirror that exactly with `Vec<f64>` rows and `i64`
//! `-1` sentinels in `AssignmentResult`.

/// `'min'` to minimise total cost, `'max'` to maximise total weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignmentDirection {
    Min,
    Max,
}

/// Result of a (possibly rectangular) bipartite assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct AssignmentResult {
    /// `rows[i] = j` iff agent i is assigned to job j; `-1` if rectangular and unmatched.
    pub rows: Vec<i64>,
    /// `cols[j] = i` iff job j is filled by agent i; `-1` if unfilled.
    pub cols: Vec<i64>,
    /// Total cost (or weight) of the optimal assignment.
    pub total: f64,
}

/// Solve a (possibly rectangular) bipartite assignment.
///
/// `matrix` is a rows × cols numeric matrix (must be rectangular). `dir` is
/// `Min` to minimise total cost (TS default), `Max` to maximise total weight.
pub fn hungarian(matrix: &[Vec<f64>], dir: AssignmentDirection) -> AssignmentResult {
    if matrix.is_empty() {
        return AssignmentResult {
            rows: Vec::new(),
            cols: Vec::new(),
            total: 0.0,
        };
    }
    let n_rows = matrix.len();
    let n_cols = matrix[0].len();
    let n = n_rows.max(n_cols);
    let sign = if dir == AssignmentDirection::Max {
        -1.0
    } else {
        1.0
    };

    // Pad to a square matrix with a constant fill (so dummy rows/cols don't
    // distort the optimal pairing). For 'max', use min-1; for 'min' use max+1.
    let fill = if dir == AssignmentDirection::Min {
        let mut mx = matrix[0][0];
        for r in matrix {
            for &v in r {
                if v > mx {
                    mx = v;
                }
            }
        }
        mx + 1.0 // dummy entries cost more than any real one
    } else {
        let mut mn = matrix[0][0];
        for r in matrix {
            for &v in r {
                if v < mn {
                    mn = v;
                }
            }
        }
        mn - 1.0 // dummy entries weigh less than any real one
    };

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = vec![0.0f64; n];
        for (j, slot) in row.iter_mut().enumerate() {
            let v = if i < n_rows && j < n_cols {
                matrix[i][j]
            } else {
                fill
            };
            *slot = sign * v;
        }
        a.push(row);
    }

    // Jonker–Volgenant via shortest-path "u, v, p" with column potentials.
    // u[i] = row potential, v[j] = column potential, p[j] = row matched to col j.
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut vv = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - vv[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    vv[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // p[j] = row assigned to column j (1-indexed). Build rows[] and cols[].
    let mut rows = vec![-1i64; n_rows];
    let mut cols = vec![-1i64; n_cols];
    let mut total = 0.0f64;
    for j in 1..=n {
        let i = p[j];
        if i == 0 {
            continue;
        }
        let ri = i - 1;
        let cj = j - 1;
        if ri < n_rows && cj < n_cols {
            rows[ri] = cj as i64;
            cols[cj] = ri as i64;
            total += matrix[ri][cj];
        }
    }

    AssignmentResult { rows, cols, total }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_min_assignment() {
        // Classic 3x3 cost matrix. Optimal min assignment:
        //   0->1 (2), 1->0 (3), 2->2 (3) => total 8? Check below.
        let m = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let res = hungarian(&m, AssignmentDirection::Min);
        // Optimal: row0->col1 (1), row1->col0 (2), row2->col2 (2) = 5.
        assert_eq!(res.total, 5.0);
        // Every row and column matched exactly once.
        let mut seen = vec![false; 3];
        for &c in &res.rows {
            assert!(c >= 0);
            assert!(!seen[c as usize]);
            seen[c as usize] = true;
        }
        // cols is the inverse mapping.
        for (i, &c) in res.rows.iter().enumerate() {
            assert_eq!(res.cols[c as usize], i as i64);
        }
    }

    #[test]
    fn identity_diagonal_is_optimal_for_max() {
        // Diagonal-dominant weight matrix: maximising picks the diagonal.
        let m = vec![
            vec![9.0, 1.0, 1.0],
            vec![1.0, 9.0, 1.0],
            vec![1.0, 1.0, 9.0],
        ];
        let res = hungarian(&m, AssignmentDirection::Max);
        assert_eq!(res.rows, vec![0, 1, 2]);
        assert_eq!(res.cols, vec![0, 1, 2]);
        assert_eq!(res.total, 27.0);
    }

    #[test]
    fn rectangular_leaves_extra_columns_unmatched() {
        // 2 agents, 3 jobs: one job stays unfilled (-1).
        let m = vec![vec![1.0, 5.0, 5.0], vec![5.0, 1.0, 5.0]];
        let res = hungarian(&m, AssignmentDirection::Min);
        assert_eq!(res.rows.len(), 2);
        assert_eq!(res.cols.len(), 3);
        assert_eq!(res.total, 2.0);
        assert_eq!(res.rows, vec![0, 1]);
        // Exactly one column unfilled.
        let unfilled = res.cols.iter().filter(|&&c| c == -1).count();
        assert_eq!(unfilled, 1);
        assert_eq!(res.cols[2], -1);
    }
}
