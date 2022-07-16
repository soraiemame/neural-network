#![allow(unused_imports)]
use rand_distr::{Normal, Distribution};
use rand::thread_rng;

pub use neural_network::matrix::Matrix;

#[derive(Debug)]
pub struct NeuralNetwork {
    // w: Vec<Vec<Vec<f64>>>,
    pub w: Vec<Matrix>, // バイアスも(最後)
    nodes: Vec<usize>
}

// const LR: f64 = 0.01;
const LR: f64 = 0.01;

impl NeuralNetwork {
    pub fn new(nodes: Vec<usize>) -> Self {
        let mut w = vec![];
        w.reserve(nodes.len() - 1);
        for i in 0..nodes.len() - 1 {
            let mut wi = Matrix::new(nodes[i] + 1, nodes[i + 1]);
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
            nodes: nodes
        }
    }
    // a[0]は埋まっている
    pub fn forward(&self,x: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut a = vec![];
        a.reserve(self.nodes.len());
        for i in 0..self.nodes.len() {
            a.push(vec![1.0;self.nodes[i]]);
        }
        for i in 0..a[0].len() {
            a[0][i] = x[i];
        }
        let mut cur = Matrix::new(1, a[0].len());
        for i in 0..a[0].len() {
            cur[0][i] = a[0][i];
        }
        cur[0].push(1.0);
        cur.w += 1;
        for i in 1..a.len() {
            let bw = &self.w[i - 1];
            cur = cur.matmul(bw);
            for j in 0..a[i].len() {
                a[i][j] = cur[0][j];
            }
            if i == a.len() - 1 { cur = cur.apply(output_activation); }
            else { cur = cur.apply(hidden_activation); }
            cur[0].push(1.0);
            cur.w += 1;
        }
        a
    }
    pub fn backward(&mut self,a: &Vec<Vec<f64>>,x: &Vec<f64>,y: &Vec<f64>,t: &Vec<f64>) -> Vec<Matrix> {
        let mut dw = vec![];
        dw.reserve(self.nodes.len() - 1);
        for i in 0..self.nodes.len() - 1 {
            let mut wi = Matrix::new(self.nodes[i] + 1, self.nodes[i + 1]);
            for j in 0..self.nodes[i] + 1 {
                for k in 0..self.nodes[i + 1] {
                    wi[j][k] = 0.0;
                }
            }
            dw.push(wi);
        }
        let mut delta = Matrix::new(a[a.len() - 1].len(), 1);
        for i in 0..a[a.len() - 1].len() {
            delta[i][0] = (y[i] - t[i]) * output_activation_delta(a[a.len() - 1][i]);
        }
        for i in (0..a.len() - 1).rev() {
            if i == 0 {
                let mut cur = Matrix::new(x.len() + 1, 1);
                for j in 0..x.len() {
                    cur[j][0] = x[j];
                }
                cur[x.len()][0] = 1.0;
                dw[i] = cur.matmul(&delta.transpose());
            }
            else {
                let mut cur = Matrix::new(a[i].len() + 1, 1);
                for j in 0..a[i].len() {
                    cur[j][0] = hidden_activation(a[i][j]);
                }
                cur[a[i].len()][0] = 1.0;
                dw[i] = cur.matmul(&delta.transpose());
                let mut w = Matrix::new(a[i].len(), a[i + 1].len());
                for j in 0..a[i].len() {
                    for k in 0..a[i + 1].len() {
                        w[j][k] = self.w[i][j][k];
                    }
                }
                cur = Matrix::new(a[i].len(),1);
                for j in 0..a[i].len() {
                    cur[j][0] = hidden_activation_delta(a[i][j]);
                }
                delta = w.matmul(&delta);
                delta = delta.hadamard(&cur);
            }
        }
        dw
    }
    pub fn train(&mut self,x: &Vec<f64>,t: &Vec<f64>) {
        let a = self.forward(x);
        let y = a[a.len() - 1].clone();
        let dw = self.backward(&a, x, &y, t);
        for i in 0..self.nodes.len() - 1 {
            for j in 0..self.w[i].h {
                for k in 0..self.w[i].w {
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
            let mut wi = Matrix::new(self.nodes[i] + 1, self.nodes[i + 1]);
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
                for k in 0..self.w[j].h {
                    for l in 0..self.w[j].w {
                        dw_avr[j][k][l] += dw[j][k][l];
                    }
                }
            }
        }
        
        for i in 0..self.nodes.len() - 1 {
            for j in 0..self.w[i].h {
                for k in 0..self.w[i].w {
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

fn hidden_activation(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))

    // if x > 0.0 { 1.0 }
    // else { 0.0 }

    // (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

fn hidden_activation_delta(x: f64) -> f64 {
    (1.0 - hidden_activation(x)) * hidden_activation(x)
    // todo!()
}

fn output_activation(x: f64) -> f64 {
    x
}

fn output_activation_delta(_x: f64) -> f64 { 1.0 }
