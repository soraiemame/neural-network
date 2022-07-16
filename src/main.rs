pub mod nn;

use nn::NeuralNetwork;

use rand::distributions::Uniform;
use rand_distr::Distribution;

use std::f64::consts::PI;

use std::time::Instant;

fn main() {
    let mut nn = NeuralNetwork::new(vec![1,10,1]);
    let n = 1000000;
    let rng = Uniform::new(0.0,PI * 2.0);
    let x = (0..n).map(|_| vec![rng.sample(&mut rand::thread_rng())]).collect::<Vec<_>>();
    let y = x.iter().map(|x| vec![x[0].sin()]).collect::<Vec<_>>();
    for i in 0..5 {
        eprintln!("sin({}) = {}",x[i][0],y[i][0]);
    }
    let start = Instant::now();
    for i in 0..n {
        nn.train(&x[i], &y[i]);
    }
    eprintln!("{:?}",Instant::now() - start);
    for _ in 0..10 {
        let tx = rng.sample(&mut rand::thread_rng());
        let ty = tx.sin();
        let res = nn.predict(&vec![tx])[0];
        println!("sin({}): {} -> loss: {}",tx,res,(ty - res).powi(2));
    }
}
