use std::marker::PhantomData;

use rand::thread_rng;
use rand_distr::{Distribution, Normal};

// pub use neural_network::matrix::Matrix;

macro_rules! mat {
    ($e:expr; $d:expr) => { vec![$e; $d] };
    ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

const LR: f64 = 0.01;

pub trait NNFunc {
    fn f(x: f64) -> f64;
    fn f_delta(x: f64) -> f64;
}

pub trait NNFuncOutput {
    fn f(a: &Vec<f64>) -> Vec<f64>;
    fn f_delta(x: f64, y: f64, t: f64) -> f64;
}

#[derive(Debug)]
pub struct NeuralNetwork<Hidden: NNFunc, Output: NNFuncOutput> {
    w: Vec<Vec<Vec<f64>>>, // バイアスも(最後)
    nodes: Vec<usize>,
    _hidden: PhantomData<Hidden>,
    _output: PhantomData<Output>,
}

impl<Hidden: NNFunc, Output: NNFuncOutput> NeuralNetwork<Hidden, Output> {
    pub fn new(nodes: Vec<usize>) -> Self
    where
        Hidden: NNFunc,
        Output: NNFuncOutput,
    {
        let mut w = vec![];
        w.reserve(nodes.len() - 1);
        for i in 0..nodes.len() - 1 {
            let mut wi = mat![0.0;nodes[i] + 1;nodes[i + 1]];
            let normal = Normal::new(0.0, 1.0 / (nodes[i] as f64).sqrt()).unwrap();
            for j in 0..nodes[i] + 1 {
                for k in 0..nodes[i + 1] {
                    wi[j][k] = normal.sample(&mut thread_rng());
                }
            }
            w.push(wi);
        }
        Self {
            w,
            nodes,
            _hidden: PhantomData,
            _output: PhantomData,
        }
    }
    unsafe fn forward(&self, x: &Vec<f64>) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut a = vec![];
        a.reserve(self.nodes.len());
        for i in 0..self.nodes.len() {
            a.push(vec![1.0; *self.nodes.get_unchecked(i)]);
        }
        // let sz0 = a.get_unchecked(0).len();
        let sz0 = *self.nodes.get_unchecked(0);
        for i in 0..sz0 {
            // a[0][i] = x[i];
            *a.get_unchecked_mut(0).get_unchecked_mut(i) = *x.get_unchecked(i);
        }
        let mut cur = vec![0.0; sz0 + 1];
        for i in 0..sz0 {
            // cur[i] = a[0][i];
            *cur.get_unchecked_mut(i) = *a.get_unchecked(0).get_unchecked(i);
        }
        // cur[a[0].len()] = 1.0;
        *cur.get_unchecked_mut(sz0) = 1.0;
        let sza = a.len();
        for i in 1..sza {
            // let bw = &self.w[i - 1];
            let bw = self.w.get_unchecked(i - 1);
            // let szi = a.get_unchecked(i).len();
            let szi = *self.nodes.get_unchecked(i);
            // let mut mul = vec![0.0; a[i].len()];
            let mut mul = vec![0.0; szi + 1];
            // for j in 0..szi {
            //     for k in 0..cur.len() {
            //         // mul[j] += cur[k] * bw[k][j];
            //         *mul.get_unchecked_mut(j) += *cur.get_unchecked(k) * *bw.get_unchecked(k).get_unchecked(j);
            //     }
            // }
            for k in 0..cur.len() {
                for j in 0..szi {
                    // mul[j] += cur[k] * bw[k][j];
                    *mul.get_unchecked_mut(j) +=
                        *cur.get_unchecked(k) * *bw.get_unchecked(k).get_unchecked(j);
                }
            }
            // mul[szi] = 1.0;
            *mul.get_unchecked_mut(szi) = 1.0;
            cur = mul;
            for j in 0..szi {
                // a[i][j] = cur[j];
                *a.get_unchecked_mut(i).get_unchecked_mut(j) = *cur.get_unchecked(j);
            }
            if i != sza - 1 {
                cur = cur.iter().map(|x| Hidden::f(*x)).collect::<Vec<_>>();
            } else {
                cur.pop();
                cur = Output::f(&cur);
            }
        }
        (a, cur)
    }
    unsafe fn backward(
        &mut self,
        a: &Vec<Vec<f64>>,
        x: &Vec<f64>,
        y: &Vec<f64>,
        t: &Vec<f64>,
    ) -> Vec<Vec<Vec<f64>>> {
        let n = self.nodes.len();
        let mut dw = vec![];
        dw.reserve(n - 1);
        for i in 0..n - 1 {
            // let mut wi = mat![0.0;self.nodes[i] + 1;self.nodes[i + 1]];
            let cur = *self.nodes.get_unchecked(i) + 1;
            let nxt = *self.nodes.get_unchecked(i + 1);
            let mut wi = mat![0.0;cur;nxt];
            for j in 0..cur {
                for k in 0..nxt {
                    // wi[j][k] = 0.0;
                    *wi.get_unchecked_mut(j).get_unchecked_mut(k) = 0.0;
                }
            }
            dw.push(wi);
        }

        let szl = a.get_unchecked(n - 1).len();
        let mut delta = vec![0.0; szl];
        for i in 0..szl {
            // delta[i] = Output::f_delta(a[a.len() - 1][i], y[i], t[i]);
            *delta.get_unchecked_mut(i) = Output::f_delta(
                *a.get_unchecked(n).get_unchecked(i),
                *y.get_unchecked(i),
                *t.get_unchecked(i),
            );
        }
        for i in (0..n - 1).rev() {
            if i == 0 {
                let szx = x.len();
                let mut cur = vec![0.0; szx + 1];
                for j in 0..szx {
                    // cur[j] = x[j];
                    *cur.get_unchecked_mut(j) = *x.get_unchecked(j);
                }
                // cur[szx] = 1.0;
                *cur.get_unchecked_mut(szx) = 1.0;
                let h = cur.len();
                let w = delta.len();
                let mut res = mat![0.0;h;w];
                for j in 0..h {
                    for k in 0..w {
                        // res[j][k] = cur[j] * delta[k];
                        *res.get_unchecked_mut(j).get_unchecked_mut(k) =
                            *cur.get_unchecked(j) * *delta.get_unchecked(k);
                    }
                }
                // dw[i] = res;
                *dw.get_unchecked_mut(i) = res;
            } else {
                let szi = *self.nodes.get_unchecked(i);
                let mut cur = vec![0.0; szi + 1];
                for j in 0..szi {
                    // cur[j] = Hidden::f(a[i][j]);
                    *cur.get_unchecked_mut(j) = Hidden::f(*a.get_unchecked(i).get_unchecked(j));
                }
                // cur[a[i].len()] = 1.0;
                *cur.get_unchecked_mut(szi) = 1.0;
                let height = cur.len();
                let width = delta.len();
                let mut res = mat![0.0;height;width];
                for j in 0..height {
                    for k in 0..width {
                        // res[j][k] = cur[j] * delta[k];
                        *res.get_unchecked_mut(j).get_unchecked_mut(k) =
                            *cur.get_unchecked(j) * *delta.get_unchecked(k);
                    }
                }
                // dw[i] = res;
                *dw.get_unchecked_mut(i) = res;
                let cur_sz = *self.nodes.get_unchecked(i);
                let nxt_sz = *self.nodes.get_unchecked(i + 1);
                let mut w = mat![0.0;cur_sz;nxt_sz];
                for j in 0..cur_sz {
                    for k in 0..nxt_sz {
                        // w[j][k] = self.w[i][j][k];
                        *w.get_unchecked_mut(j).get_unchecked_mut(k) =
                            *self.w.get_unchecked(i).get_unchecked(j).get_unchecked(k);
                    }
                }
                let mut cur = vec![0.0; cur_sz];
                for j in 0..cur_sz {
                    // cur[j] = Hidden::f_delta(a[i][j]);
                    *cur.get_unchecked_mut(j) =
                        Hidden::f_delta(*a.get_unchecked(i).get_unchecked(j));
                }
                let mut next_delta = vec![0.0; cur_sz];
                for j in 0..cur_sz {
                    for k in 0..nxt_sz {
                        // next_delta[j] += delta[k] * w[j][k];
                        *next_delta.get_unchecked_mut(j) +=
                            *delta.get_unchecked(k) * *w.get_unchecked(j).get_unchecked(k);
                    }
                }
                delta = next_delta;
                for j in 0..cur_sz {
                    // delta[j] *= cur[j];
                    *delta.get_unchecked_mut(j) *= *cur.get_unchecked(j);
                }
            }
        }
        dw
    }
    pub fn train_single(&mut self, x: &Vec<f64>, t: &Vec<f64>) {
        assert_eq!(self.nodes[0],x.len());
        assert_eq!(self.nodes[self.nodes.len() - 1],t.len());
        let (a, y) = unsafe { self.forward(x) };
        let dw = unsafe { self.backward(&a, x, &y, t) };

        unsafe {
            for i in 0..self.nodes.len() - 1 {
                for j in 0..self.w[i].len() {
                    let width = self.w[i][0].len();
                    for k in 0..width {
                        // self.w[i][j][k] -= LR * dw[i][j][k];
                        *self.w.get_unchecked_mut(i).get_unchecked_mut(j).get_unchecked_mut(k) -=
                            LR * *dw.get_unchecked(i).get_unchecked(j).get_unchecked(k);
                    }
                }
            }
        }
    }
    pub fn train_mul(&mut self, x: &Vec<Vec<f64>>, t: &Vec<Vec<f64>>) {
        assert_eq!(x.len(), t.len());
        let mut dw_avr = vec![];
        dw_avr.reserve(self.nodes.len() - 1);
        for i in 0..self.nodes.len() - 1 {
            let mut wi = mat![0.0;self.nodes[i] + 1;self.nodes[i + 1]];
            for j in 0..self.nodes[i] + 1 {
                for k in 0..self.nodes[i + 1] {
                    wi[j][k] = 0.0;
                }
            }
            dw_avr.push(wi);
        }
        let n = x.len();
        for i in 0..n {
            let (a, y) = unsafe { self.forward(&x[i]) };
            let dw = unsafe { self.backward(&a, &x[i], &y, &t[i]) };
            for j in 0..self.nodes.len() - 1 {
                for k in 0..self.w[j].len() {
                    for l in 0..self.w[j][0].len() {
                        dw_avr[j][k][l] += dw[j][k][l];
                    }
                }
            }
        }

        for i in 0..self.nodes.len() - 1 {
            for j in 0..self.w[i].len() {
                for k in 0..self.w[i][0].len() {
                    dw_avr[i][j][k] /= n as f64;
                    self.w[i][j][k] -= LR * dw_avr[i][j][k];
                }
            }
        }
    }
    pub fn predict(&self, x: &Vec<f64>) -> Vec<f64> {
        let (_, res) = unsafe { self.forward(x) };
        res
    }
}
