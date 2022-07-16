use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub h: usize,
    pub w: usize,
    pub dat: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(h: usize, w: usize) -> Self {
        Self {
            h: h,
            w: w,
            dat: vec![vec![0.0; w]; h],
        }
    }
    pub fn matmul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.w, rhs.h);
        let mut res = Matrix::new(self.h, rhs.w);
        for i in 0..self.h {
            for j in 0..rhs.w {
                for k in 0..self.w {
                    res[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        res
    }
    pub fn hadamard(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.h, rhs.h);
        assert_eq!(self.w, rhs.w);
        let mut res = Matrix::new(self.h, self.w);
        for i in 0..self.h {
            for j in 0..self.w {
                res[i][j] = self[i][j] * rhs[i][j];
            }
        }
        res
    }
    pub fn transpose(&self) -> Matrix {
        let mut res = Matrix::new(self.w, self.h);
        for i in 0..self.h {
            for j in 0..self.w {
                res[j][i] = self[i][j];
            }
        }
        res
    }
    pub fn apply<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let mut res = Matrix::new(self.h, self.w);
        for i in 0..self.h {
            for j in 0..self.w {
                res[i][j] = f(self[i][j]);
            }
        }
        res
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.dat[index]
    }
}
impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dat[index]
    }
}
