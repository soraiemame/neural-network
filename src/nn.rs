
use std::marker::PhantomData;

use rand_distr::{Normal, Distribution};
use rand::thread_rng;

// pub use neural_network::matrix::Matrix;

macro_rules! mat {
    ($e:expr; $d:expr) => { vec![$e; $d] };
    ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

// const LR: f64 = 0.01;
const LR: f64 = 0.01;

pub trait NNFunc {
    fn f(x: f64) -> f64;
    fn f_delta(x: f64) -> f64;
}

#[derive(Debug)]
pub struct NeuralNetwork<Hidden: NNFunc,Output: NNFunc> {
    w: Vec<Vec<Vec<f64>>>, // バイアスも(最後)
    nodes: Vec<usize>,
    _hidden: PhantomData<Hidden>,
    _output: PhantomData<Output>
}

impl<Hidden: NNFunc,Output: NNFunc> NeuralNetwork<Hidden,Output> {
    pub fn new(nodes: Vec<usize>) -> Self
    where Hidden: NNFunc,
          Output: NNFunc
    {
        let mut w = vec![];
        w.reserve(nodes.len() - 1);
        for i in 0..nodes.len() - 1 {
            // let mut wi = Matrix::new(nodes[i] + 1, nodes[i + 1]);
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
            w: w,
            nodes: nodes,
            _hidden: PhantomData,
            _output: PhantomData
        }
    }
    // a[0]は埋まっている
    fn forward(&self,x: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut a = vec![];
        a.reserve(self.nodes.len());
        for i in 0..self.nodes.len() {
            a.push(vec![1.0;self.nodes[i]]);
        }
        for i in 0..a[0].len() {
            a[0][i] = x[i];
        }
        let mut cur = vec![0.0;a[0].len() + 1];
        for i in 0..a[0].len() {
            cur[i] = a[0][i];
        }
        cur[a[0].len()] = 1.0;
        for i in 1..a.len() {
            let bw = &self.w[i - 1];
            let mut mul = vec![0.0;a[i].len()];
            for j in 0..a[i].len() {
                for k in 0..cur.len() {
                    mul[j] += cur[k] * bw[k][j];
                }
            }
            cur = mul;
            for j in 0..a[i].len() {
                a[i][j] = cur[j];
            }
            if i == a.len() - 1 { cur = cur.iter().map(|x| Output::f(*x)).collect::<Vec<_>>(); }
            else { cur = cur.iter().map(|x| Hidden::f(*x)).collect::<Vec<_>>(); }
            cur.push(1.0);
        }
        a
    }
    fn backward(&mut self,a: &Vec<Vec<f64>>,x: &Vec<f64>,y: &Vec<f64>,t: &Vec<f64>) -> Vec<Vec<Vec<f64>>> {
        let mut dw = vec![];
        dw.reserve(self.nodes.len() - 1);
        for i in 0..self.nodes.len() - 1 {
            let mut wi = mat![0.0;self.nodes[i] + 1;self.nodes[i + 1]];
            for j in 0..self.nodes[i] + 1 {
                for k in 0..self.nodes[i + 1] {
                    wi[j][k] = 0.0;
                }
            }
            dw.push(wi);
        }

        let mut delta = vec![0.0;a[a.len() - 1].len()];
        for i in 0..a[a.len() - 1].len() {
            delta[i] = (y[i] - t[i]) * Output::f_delta(a[a.len() - 1][i]);
        }
        for i in (0..a.len() - 1).rev() {
            if i == 0 {
                let mut cur = vec![0.0;x.len() + 1];
                for j in 0..x.len() {
                    cur[j] = x[j];
                }
                cur[x.len()] = 1.0;
                let mut res = mat![0.0;cur.len();delta.len()];
                for j in 0..res.len() {
                    for k in 0..res[0].len() {
                        res[j][k] = cur[j] * delta[k];
                    }
                }
                dw[i] = res;
            }
            else {
                let mut cur = vec![0.0;a[i].len() + 1];
                for j in 0..a[i].len() {
                    cur[j] = Hidden::f(a[i][j]);
                }
                cur[a[i].len()] = 1.0;
                let mut res = mat![0.0;cur.len();delta.len()];
                for j in 0..res.len() {
                    for k in 0..res[0].len() {
                        res[j][k] = cur[j] * delta[k];
                    }
                }
                dw[i] = res;
                let mut w = mat![0.0;a[i].len();a[i + 1].len()];
                for j in 0..a[i].len() {
                    for k in 0..a[i + 1].len() {
                        w[j][k] = self.w[i][j][k];
                    }
                }
                let mut cur = vec![0.0;a[i].len()];
                for j in 0..a[i].len() {
                    cur[j] = Hidden::f_delta(a[i][j]);
                }
                let mut next_delta = vec![0.0;w.len()];
                for j in 0..w.len() {
                    for k in 0..w[0].len() {
                        next_delta[j] += delta[k] * w[j][k];
                    }
                }
                delta = next_delta;
                for j in 0..w.len() {
                    delta[j] *= cur[j];
                }
            }
        }
        dw
    }
    pub fn train(&mut self,x: &Vec<f64>,t: &Vec<f64>) {
        let a = self.forward(x);
        let y = a[a.len() - 1].clone();
        let dw = self.backward(&a, x, &y, t);
        for i in 0..self.nodes.len() - 1 {
            for j in 0..self.w[i].len() {
                for k in 0..self.w[i][0].len() {
                    self.w[i][j][k] -= LR * dw[i][j][k];
                }
            }
        }
    }
    pub fn train_mul(&mut self,x: &Vec<Vec<f64>>,t: &Vec<Vec<f64>>) {
        assert_eq!(x.len(),t.len());
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
            let a = self.forward(&x[i]);
            let y = a[a.len() - 1].clone();
            let dw = self.backward(&a, &x[i], &y, &t[i]);
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
    pub fn predict(&self,x: &Vec<f64>) -> Vec<f64> {
        let res = self.forward(x);
        res[self.nodes.len() - 1].clone()
    }
}
