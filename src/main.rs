pub mod nn;

use nn::NeuralNetwork;

use proconio::input;

use rand::distributions::Uniform;
use rand_distr::Distribution;

use std::f64::consts::PI;

fn main() {
    // input! {
    //     n: usize,p: usize,m: usize,N: usize,
    //     v: [[f64;p];n],
    //     a: [f64;p],
    //     w: [[f64;m];p],
    //     b: [f64;m],
    //     x: [[f64;n];N],
    //     t: [[f64;m];N],
    // }
    // let mut nn = NeuralNetwork::new(vec![n,p,m]);
    // // eprintln!("{:?}",nn);
    // for i in 0..n + 1 {
    //     for j in 0..p {
    //         if i != n {
    //             nn.w[0][i][j] = v[i][j]
    //         }
    //         else {
    //             nn.w[0][i][j] = a[j];
    //         }
    //     }
    // }
    // for i in 0..p + 1 {
    //     for j in 0..m {
    //         if i != p {
    //             nn.w[1][i][j] = w[i][j];
    //         }
    //         else {
    //             nn.w[1][i][j] = b[j];
    //         }
    //     }
    // }
    // // nn.train(&x[0], &t[0]);
    // nn.train_mul(&x, &t);
    // eprintln!("-------------------------------------------------------------------------------------------------------------------------");
    // let a = nn.forward(&x[0]);
    // eprintln!("{:?}",a);
    // // eprintln!("{:?}",nn);
    // // let a = nn.forward(&x[0]);
    // // eprintln!("a: {:?}",a);
    // // eprintln!("x: {:?}",x[0]);
    // // eprintln!("y: {:?}",a[a.len() - 1]);
    // // eprintln!("t: {:?}",t[0]);
    // // nn.backward(&a, &x[0], &a[a.len() - 1], &t[0]);
    let mut nn = NeuralNetwork::new(vec![1,10,1]);
    let n = 1000000;
    let rng = Uniform::new(0.0,PI * 2.0);
    let x = (0..n).map(|_| vec![rng.sample(&mut rand::thread_rng())]).collect::<Vec<_>>();
    let y = x.iter().map(|x| vec![x[0].sin()]).collect::<Vec<_>>();
    for i in 0..5 {
        eprintln!("sin({}) = {}",x[i][0],y[i][0]);
    }
    // for i in 0..10000 {
    //     let mut sx = vec![];
    //     let mut sy = vec![];
    //     for j in 0..10 {
    //         sx.push(x[i * 10 + j].clone());
    //         sy.push(y[i * 10 + j].clone());
    //     }
    //     nn.train_mul(&sx, &sy);
    // }
    for i in 0..n {
        nn.train(&x[i], &y[i]);
    }
    // println!("PI / 2: {}, PI / 3: {}, PI / 4: {}",nn.forward(&x))
    // eprintln!("{:?}",nn.forward(&vec![PI / 2.0]));
    // eprintln!("{:?}",nn.forward(&vec![PI / 3.0]));
    // eprintln!("{:?}",nn.forward(&vec![PI / 4.0]));
    for _ in 0..10 {
        let tx = rng.sample(&mut rand::thread_rng());
        let ty = tx.sin();
        let res = nn.forward(&vec![tx])[2][0];
        println!("sin({}): {} -> loss: {}",tx,res,(ty - res).powi(2));
    }
}
