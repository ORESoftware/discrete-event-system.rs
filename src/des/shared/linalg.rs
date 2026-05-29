//! Port of `src/des/shared/linalg.ts`.
//!
//! Dependency-free dense linear-algebra toolkit. The TypeScript exposes `Vec`
//! and `Mat`; since `Vec` is a std type in Rust we alias them as `Vector` and
//! `Matrix`. `LinAlg` / `VecOps` static methods become associated functions;
//! the stateful algorithms (`MatrixInverse`, `LinearSystem`, `MatrixRank`,
//! `SymmetricEigen`) become structs that own their scratch state.
//!
//! Shape mismatches and singular matrices `panic!` here (they were `throw` in
//! TS — programmer-error / invariant violations).

pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vec<f64>>;

/// Stateless matrix arithmetic (associated functions).
pub struct LinAlg;

impl LinAlg {
    /// n×n identity.
    pub fn identity(n: usize) -> Matrix {
        let mut out = vec![vec![0.0; n]; n];
        for (i, row) in out.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        out
    }

    /// r×c zero matrix.
    pub fn zeros(r: usize, c: usize) -> Matrix {
        vec![vec![0.0; c]; r]
    }

    /// Deep copy.
    pub fn copy(m: &Matrix) -> Matrix {
        m.clone()
    }

    /// Number of rows.
    pub fn rows(m: &Matrix) -> usize {
        m.len()
    }

    /// Number of columns (0 for an empty matrix).
    pub fn cols(m: &Matrix) -> usize {
        if m.is_empty() {
            0
        } else {
            m[0].len()
        }
    }

    /// Transpose.
    pub fn transpose(m: &Matrix) -> Matrix {
        let r = Self::rows(m);
        let c = Self::cols(m);
        let mut out = vec![vec![0.0; r]; c];
        for i in 0..r {
            for j in 0..c {
                out[j][i] = m[i][j];
            }
        }
        out
    }

    /// Matrix product A·B. Panics on a shape mismatch.
    pub fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
        let ra = Self::rows(a);
        let ca = Self::cols(a);
        let rb = Self::rows(b);
        let cb = Self::cols(b);
        if ca != rb {
            panic!("LinAlg::mat_mul: shape mismatch {ra}x{ca} · {rb}x{cb}");
        }
        let mut out = Self::zeros(ra, cb);
        for i in 0..ra {
            for k in 0..ca {
                let av = a[i][k];
                if av == 0.0 {
                    continue;
                }
                for j in 0..cb {
                    out[i][j] += av * b[k][j];
                }
            }
        }
        out
    }

    /// Matrix·vector M·v.
    pub fn mat_vec(m: &Matrix, v: &[f64]) -> Vector {
        let r = Self::rows(m);
        let c = Self::cols(m);
        if c != v.len() {
            panic!("LinAlg::mat_vec: shape mismatch {r}x{c} · {}", v.len());
        }
        let mut out = vec![0.0; r];
        for i in 0..r {
            let mut acc = 0.0;
            for j in 0..c {
                acc += m[i][j] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// A + B.
    pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
        let r = Self::rows(a);
        let c = Self::cols(a);
        let mut out = Self::zeros(r, c);
        for i in 0..r {
            for j in 0..c {
                out[i][j] = a[i][j] + b[i][j];
            }
        }
        out
    }

    /// A − B.
    pub fn sub(a: &Matrix, b: &Matrix) -> Matrix {
        let r = Self::rows(a);
        let c = Self::cols(a);
        let mut out = Self::zeros(r, c);
        for i in 0..r {
            for j in 0..c {
                out[i][j] = a[i][j] - b[i][j];
            }
        }
        out
    }

    /// s·M.
    pub fn scale(m: &Matrix, s: f64) -> Matrix {
        m.iter().map(|row| row.iter().map(|x| x * s).collect()).collect()
    }

    /// A^k via repeated multiplication (k ≥ 0; k = 0 → identity).
    pub fn power(a: &Matrix, k: usize) -> Matrix {
        let n = Self::rows(a);
        let mut acc = Self::identity(n);
        for _ in 0..k {
            acc = Self::mat_mul(&acc, a);
        }
        acc
    }

    /// Horizontal block concatenation [A | B | …]: same row count.
    pub fn hstack(blocks: &[Matrix]) -> Matrix {
        if blocks.is_empty() {
            panic!("LinAlg::hstack: no blocks");
        }
        let r = Self::rows(&blocks[0]);
        for b in blocks {
            if Self::rows(b) != r {
                panic!("LinAlg::hstack: row-count mismatch");
            }
        }
        let mut out = vec![Vec::new(); r];
        for b in blocks {
            for i in 0..r {
                out[i].extend_from_slice(&b[i]);
            }
        }
        out
    }

    /// Vertical block concatenation [A; B; …]: same column count.
    pub fn vstack(blocks: &[Matrix]) -> Matrix {
        if blocks.is_empty() {
            panic!("LinAlg::vstack: no blocks");
        }
        let c = Self::cols(&blocks[0]);
        let mut out: Matrix = Vec::new();
        for b in blocks {
            if Self::cols(b) != c {
                panic!("LinAlg::vstack: column-count mismatch");
            }
            for row in b {
                out.push(row.clone());
            }
        }
        out
    }

    /// Largest absolute entry.
    pub fn max_abs(m: &Matrix) -> f64 {
        let mut mx = 0.0_f64;
        for row in m {
            for x in row {
                let a = x.abs();
                if a > mx {
                    mx = a;
                }
            }
        }
        mx
    }

    /// Numeric rank via `MatrixRank`.
    pub fn rank(m: &Matrix, tol: Option<f64>) -> usize {
        MatrixRank::new(m, tol).rank()
    }
}

/// Stateless vector arithmetic.
pub struct VecOps;

impl VecOps {
    /// Dot product aᵀb.
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        s
    }

    /// Euclidean (L2) norm ‖v‖₂.
    pub fn norm2(v: &[f64]) -> f64 {
        Self::dot(v, v).sqrt()
    }

    /// a + b.
    pub fn add(a: &[f64], b: &[f64]) -> Vector {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// a − b.
    pub fn sub(a: &[f64], b: &[f64]) -> Vector {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }

    /// s·v.
    pub fn scale(v: &[f64], s: f64) -> Vector {
        v.iter().map(|x| x * s).collect()
    }

    /// y + s·x (BLAS axpy).
    pub fn axpy(s: f64, x: &[f64], y: &[f64]) -> Vector {
        y.iter().zip(x).map(|(yi, xi)| yi + s * xi).collect()
    }

    /// Zero vector of length n.
    pub fn zeros(n: usize) -> Vector {
        vec![0.0; n]
    }
}

/// Dense matrix inverse via Gauss–Jordan elimination with partial pivoting.
/// Panics if the matrix is singular to the given tolerance.
pub struct MatrixInverse {
    n: usize,
    aug: Matrix,
    tol: f64,
    result: Option<Matrix>,
}

impl MatrixInverse {
    pub fn new(m: &Matrix, tol: Option<f64>) -> Self {
        let n = LinAlg::rows(m);
        if LinAlg::cols(m) != n {
            panic!("MatrixInverse: matrix must be square");
        }
        let tol = tol.unwrap_or_else(|| (1.0_f64).max(LinAlg::max_abs(m)) * n as f64 * 1e-14);
        let ident = LinAlg::identity(n);
        let aug: Matrix = m
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r.extend_from_slice(&ident[i]);
                r
            })
            .collect();
        MatrixInverse {
            n,
            aug,
            tol,
            result: None,
        }
    }

    pub fn inverse(&mut self) -> Matrix {
        if let Some(r) = &self.result {
            return r.clone();
        }
        let n = self.n;
        let a = &mut self.aug;
        for col in 0..n {
            let mut best = col;
            for i in (col + 1)..n {
                if a[i][col].abs() > a[best][col].abs() {
                    best = i;
                }
            }
            if a[best][col].abs() <= self.tol {
                panic!("MatrixInverse: matrix is singular");
            }
            if best != col {
                a.swap(best, col);
            }
            let piv = a[col][col];
            for j in 0..(2 * n) {
                a[col][j] /= piv;
            }
            for i in 0..n {
                if i == col {
                    continue;
                }
                let f = a[i][col];
                if f == 0.0 {
                    continue;
                }
                for j in 0..(2 * n) {
                    a[i][j] -= f * a[col][j];
                }
            }
        }
        let result: Matrix = a.iter().map(|row| row[n..].to_vec()).collect();
        self.result = Some(result.clone());
        result
    }

    /// Solve M·X = B using the computed inverse.
    pub fn solve(&mut self, b: &Matrix) -> Matrix {
        LinAlg::mat_mul(&self.inverse(), b)
    }
}

/// Solve a single dense system A·x = b via Gaussian elimination with partial
/// pivoting. Panics if the matrix is singular to the given tolerance.
pub struct LinearSystem {
    n: usize,
    a: Matrix,
    b: Vector,
    tol: f64,
}

impl LinearSystem {
    pub fn new(a: &Matrix, b: &[f64], tol: f64) -> Self {
        LinearSystem {
            n: b.len(),
            a: a.clone(),
            b: b.to_vec(),
            tol,
        }
    }

    /// Solve, panicking on a singular matrix.
    pub fn solve(&self) -> Vector {
        self.try_solve().expect("LinearSystem: singular matrix")
    }

    /// Solve, returning `None` on a singular matrix (TS `try { solve } catch`).
    pub fn try_solve(&self) -> Option<Vector> {
        let n = self.n;
        let mut m = self.a.clone();
        let mut x = self.b.clone();
        for i in 0..n {
            let mut pivot = i;
            for k in (i + 1)..n {
                if m[k][i].abs() > m[pivot][i].abs() {
                    pivot = k;
                }
            }
            if m[pivot][i].abs() < self.tol {
                return None;
            }
            if pivot != i {
                m.swap(i, pivot);
                x.swap(i, pivot);
            }
            for k in (i + 1)..n {
                let f = m[k][i] / m[i][i];
                for j in i..n {
                    m[k][j] -= f * m[i][j];
                }
                x[k] -= f * x[i];
            }
        }
        let mut y = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in (i + 1)..n {
                s -= m[i][j] * y[j];
            }
            y[i] = s / m[i][i];
        }
        Some(y)
    }
}

/// Numeric-rank computation (row-reduction with partial pivoting).
pub struct MatrixRank {
    work: Matrix,
    tol: f64,
    rank_value: Option<usize>,
}

impl MatrixRank {
    pub fn new(m: &Matrix, tol: Option<f64>) -> Self {
        let r = LinAlg::rows(m);
        let c = LinAlg::cols(m);
        let scale = (1.0_f64).max(LinAlg::max_abs(m));
        let tol = tol.unwrap_or_else(|| (r.max(c)) as f64 * scale * 1e-12);
        MatrixRank {
            work: m.clone(),
            tol,
            rank_value: None,
        }
    }

    pub fn rank(&mut self) -> usize {
        if let Some(rv) = self.rank_value {
            return rv;
        }
        let a = &mut self.work;
        let r = a.len();
        let c = if r == 0 { 0 } else { a[0].len() };
        let mut pivot_row = 0usize;
        let mut col = 0usize;
        while col < c && pivot_row < r {
            let mut best = pivot_row;
            for i in (pivot_row + 1)..r {
                if a[i][col].abs() > a[best][col].abs() {
                    best = i;
                }
            }
            if a[best][col].abs() <= self.tol {
                col += 1;
                continue;
            }
            if best != pivot_row {
                a.swap(best, pivot_row);
            }
            let piv = a[pivot_row][col];
            for i in 0..r {
                if i == pivot_row {
                    continue;
                }
                let factor = a[i][col] / piv;
                if factor == 0.0 {
                    continue;
                }
                for j in col..c {
                    a[i][j] -= factor * a[pivot_row][j];
                }
            }
            pivot_row += 1;
            col += 1;
        }
        self.rank_value = Some(pivot_row);
        pivot_row
    }

    pub fn is_full_rank(&mut self, n: usize) -> bool {
        self.rank() == n
    }
}

/// Eigen-decomposition of a SYMMETRIC matrix via cyclic Jacobi rotations.
/// Eigenvalues returned ASCENDING; eigenvectors are the columns of `vectors()`.
pub struct SymmetricEigen {
    n: usize,
    source: Matrix,
    sweeps: usize,
    vals: Option<Vector>,
    vecs: Option<Matrix>,
}

impl SymmetricEigen {
    pub fn new(m: &Matrix, sweeps: usize) -> Self {
        let n = LinAlg::rows(m);
        if LinAlg::cols(m) != n {
            panic!("SymmetricEigen: matrix must be square");
        }
        SymmetricEigen {
            n,
            source: LinAlg::copy(m),
            sweeps,
            vals: None,
            vecs: None,
        }
    }

    fn compute(&mut self) {
        let n = self.n;
        let mut a = LinAlg::copy(&self.source);
        let mut v = LinAlg::identity(n);
        for _ in 0..self.sweeps {
            let mut off = 0.0;
            for p in 0..n {
                for q in (p + 1)..n {
                    off += a[p][q] * a[p][q];
                }
            }
            if off < 1e-30 {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    if a[p][q].abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let sign = if theta == 0.0 { 1.0 } else { theta.signum() };
                    let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[k][p];
                        let akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[p][k];
                        let aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for k in 0..n {
                        let vkp = v[k][p];
                        let vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| a[i][i].partial_cmp(&a[j][j]).unwrap());
        self.vals = Some(idx.iter().map(|&i| a[i][i]).collect());
        self.vecs = Some((0..n).map(|r| idx.iter().map(|&i| v[r][i]).collect()).collect());
    }

    pub fn values(&mut self) -> Vector {
        if self.vals.is_none() {
            self.compute();
        }
        self.vals.clone().unwrap()
    }

    pub fn vectors(&mut self) -> Matrix {
        if self.vecs.is_none() {
            self.compute();
        }
        LinAlg::copy(self.vecs.as_ref().unwrap())
    }

    pub fn min_eigenvalue(&mut self) -> f64 {
        (0.0_f64).max(self.values()[0])
    }

    pub fn max_eigenvalue(&mut self) -> f64 {
        let v = self.values();
        v[v.len() - 1]
    }

    pub fn condition_number(&mut self) -> f64 {
        let lo = self.min_eigenvalue();
        if lo <= 0.0 {
            f64::INFINITY
        } else {
            self.max_eigenvalue() / lo
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_matmul() {
        let i = LinAlg::identity(3);
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 10.0]];
        assert_eq!(LinAlg::mat_mul(&i, &m), m);
    }

    #[test]
    fn linear_system_solves() {
        // [[2,1],[1,3]] x = [3,5]  => x = [0.8, 1.4]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let x = LinearSystem::new(&a, &[3.0, 5.0], 1e-15).solve();
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn inverse_roundtrip() {
        let m = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = MatrixInverse::new(&m, None).inverse();
        let prod = LinAlg::mat_mul(&m, &inv);
        assert!((prod[0][0] - 1.0).abs() < 1e-12);
        assert!((prod[1][1] - 1.0).abs() < 1e-12);
        assert!(prod[0][1].abs() < 1e-12);
    }

    #[test]
    fn rank_of_singular() {
        let m = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(LinAlg::rank(&m, None), 1);
    }

    #[test]
    fn vecops_basics() {
        assert_eq!(VecOps::dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert!((VecOps::norm2(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_eigen_diag() {
        let m = vec![vec![2.0, 0.0], vec![0.0, 5.0]];
        let mut e = SymmetricEigen::new(&m, 100);
        let vals = e.values();
        assert!((vals[0] - 2.0).abs() < 1e-9);
        assert!((vals[1] - 5.0).abs() < 1e-9);
    }
}
