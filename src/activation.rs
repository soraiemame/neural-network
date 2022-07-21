use crate::{NNFunc,NNFuncOutput};

pub struct Sigmoid;
impl NNFunc for Sigmoid {
    fn f(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }
    fn f_delta(x: f64) -> f64 {
        (1.0 - Self::f(x)) * Self::f(x)
    }
}

pub struct Relu;
impl NNFunc for Relu {
    fn f(x: f64) -> f64 {
        if x < 0.0 { 0.0 }
        else { x }
    }
    fn f_delta(x: f64) -> f64 {
        if x < 0.0 { 0.0 }
        else { 1.0 }
    }
}

pub struct LeakyRelu;
impl NNFunc for LeakyRelu {
    fn f(x: f64) -> f64 {
        if x < 0.0 { x * 0.01 }
        else { x }
    }
    fn f_delta(x: f64) -> f64 {
        if x < 0.0 { 0.01 }
        else { 1.0 }
    }
}

pub struct Id;
impl NNFunc for Id {
    fn f(x: f64) -> f64 {
        x
    }
    fn f_delta(_x: f64) -> f64 {
        1.0
    }
}
impl NNFuncOutput for Id {
    fn f(a: &Vec<f64>) -> Vec<f64> {
        a.clone()
    }
    fn f_delta(_x: f64,y: f64, t: f64) -> f64 {
        y - t
    }
}

pub struct SoftMax;
impl NNFuncOutput for SoftMax {
    fn f(a: &Vec<f64>) -> Vec<f64> {
        let fa = a.iter().map(|x| x.exp()).collect::<Vec<_>>();
        let sum = fa.iter().sum::<f64>();
        fa.iter().map(|x| x / sum).collect::<Vec<_>>()
    }
    fn f_delta(_x: f64,y: f64, t: f64) -> f64 {
        y - t
    }
}