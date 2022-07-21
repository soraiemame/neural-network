#![allow(unused_imports)]
#![allow(dead_code)]

pub mod nn;
pub mod activation;

use nn::NeuralNetwork;

use rand::{SeedableRng, Rng, RngCore};
use rand::distributions::Uniform;
use rand_distr::Distribution;

use std::f64::consts::PI;

use std::time::Instant;

use nn::{NNFunc,NNFuncOutput};
use activation::{Id,Relu,LeakyRelu,Sigmoid,SoftMax};

use dotenv::dotenv;

fn train_sin() {
    let mut nn = NeuralNetwork::<Sigmoid, Id>::new(vec![1, 10, 1]);
    let n = 1000000;
    // let n = 1000;
    let rng = Uniform::new(0.0, PI * 2.0);

    // let rng = rand::rngs::StdRng::from_entropy();
    let x = (0..n)
        .map(|_| vec![rng.sample(&mut rand::thread_rng())])
        .collect::<Vec<_>>();
    let y = x.iter().map(|x| vec![x[0].sin()]).collect::<Vec<_>>();
    for i in 0..5 {
        eprintln!("{:?} -> {}", x[i], y[i][0]);
    }
    let start = Instant::now();
    for i in 0..n {
        nn.train_single(&x[i],&y[i]);
    }
    eprintln!("{} cases, time: {:?}", n, Instant::now() - start);
    for _ in 0..10 {
        let tx = rng.sample(&mut rand::thread_rng());
        // let tx = vec![];
        let ty = tx.sin();
        let res = nn.predict(&vec![tx])[0];
        println!("{}: {} -> loss: {}", tx, res, (ty - res).powi(2));
    }
}

fn train_softmax() {
    let mut nn = NeuralNetwork::<Sigmoid,SoftMax>::new(vec![1,2]);
    let n = 10000;
    let rng = Uniform::new(-1.0,1.0);
    let x = (0..n).map(|_| vec![rng.sample(&mut rand::thread_rng())]).collect::<Vec<_>>();
    let y = x.iter()
        .map(|x| vec![if x[0] < 0.0 { 1.0 } else { 0.0 },if x[0] > 0.0 { 1.0 } else { 0.0 }])
        .collect::<Vec<_>>();
    for i in 0..5 {
        eprintln!("{:?} -> {:?}",x[i],y[i]);
    }
    let start = Instant::now();
    for i in 0..n {
        nn.train_single(&x[i],&y[i]);
    }
    eprintln!("{} cases, time: {:?}", n, Instant::now() - start);
    for _ in 0..10 {
        let tx = rng.sample(&mut rand::thread_rng());
        // let tx = vec![];
        // let ty = tx.sin();
        let res = nn.predict(&vec![tx]);
        println!("{} -> {:?}", tx, res);
    }
}

fn train_mnist() {
    use std::path::Path;
    use std::fs::File;
    use std::io::prelude::*;
    use std::env;
    dotenv().ok();
    let train_path = env::var("TRAIN_DATASET_PATH").unwrap();
    let train_path = Path::new(&train_path);
    let train_display = train_path.display();
    let mut train_file = match File::open(&train_path) {
        Err(why) => panic!("couldn't open {}: {}", train_display, why),
        Ok(file) => file,
    };
    let mut s = String::new();
    match train_file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", train_display, why),
        Ok(_) => println!("successfully read {}",train_display),
    }
    let s = s.split_whitespace().collect::<Vec<_>>();
    let train = s.iter()
        .map(|x| x.split(",").map(|y| y.parse::<u8>().unwrap()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    println!("data num: {}, one data len: {}",s.len(),s[0].len());
    let mut nn = NeuralNetwork::<Sigmoid,SoftMax>::new(vec![784,64,64,10]);
    let train_x = train.iter()
        .map(|x|
            x[1..].to_owned().iter().map(|x| (*x as f64) / 256.0
        ).collect::<Vec<_>>()).collect::<Vec<_>>();
    let train_y = train.iter()
        .map(|x| {
            let mut res = vec![0.0;10];
            res[x[0] as usize] = 1.0;
            res
        })
        .collect::<Vec<_>>();
    let start = Instant::now();
    let case_len = s.len();
    let n = case_len;
    for i in 0..n {
        if i % 100 == 0 { eprintln!("trained: {}",i); }
        nn.train_single(&train_x[i],&train_y[i]);
    }
    eprintln!("{} cases, time: {:?}", n, Instant::now() - start);
    let test_path = env::var("TEST_DATASET_PATH").unwrap();
    let test_path = Path::new(&test_path);
    let test_display = test_path.display();
    let mut test_file = match File::open(&test_path) {
        Err(why) => panic!("couldn't open {}: {}", test_display, why),
        Ok(file) => file,
    };
    let mut s = String::new();
    match test_file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", test_display, why),
        Ok(_) => println!("successfully read {}",test_display),
    }
    let s = s.split_whitespace().collect::<Vec<_>>();
    let test = s.iter()
        .map(|x| x.split(",").map(|y| y.parse::<u8>().unwrap()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let test_x = test.iter()
        .map(|x|
            x[1..].to_owned().iter().map(|x| (*x as f64) / 256.0
        ).collect::<Vec<_>>()).collect::<Vec<_>>();
    let test_y = test.iter()
        .map(|x| {
            let mut res = vec![0.0;10];
            res[x[0] as usize] = 1.0;
            res
        })
        .collect::<Vec<_>>();
    let tests = test.len();
    println!("test len: {}",tests);
    let mut correct = 0;
    for i in 0..tests {
        let x = &test_x[i];
        let y = &test_y[i];
        let pre = nn.predict(x);
        let mut pre_max: f64 = 0.0;
        for j in 0..10 {
            pre_max = pre_max.max(pre[j]);
        }
        let mut y_num = 10;
        let mut pre_num = 10;
        for j in 0..10 {
            if y[j] == 1.0 { y_num = j; }
        }
        for j in 0..10 {
            if pre[j] == pre_max { pre_num = j; }
        }
        assert_ne!(y_num,10);
        assert_ne!(pre_num,10);
        if y_num == pre_num { correct += 1; }
    }
    println!("testcases: {}, correct: {}, accuray: {}",tests,correct,correct as f64 / tests as f64);
}

fn main() {
    // train_softmax();
    train_mnist();
}
