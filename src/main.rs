use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::{set_server_key};
use tfhe::{ConfigBuilder, ClientKey, CompressedServerKey, CudaServerKey, FheUint16};
use std::time::Instant;
use crate::network::*;
use crate::plain_layers::{dense_plain, PlainDenseLayer};
use rand::Rng;
use half::f16;
use tfhe::prelude::*;

mod encrypted_layers;
mod plain_layers;
mod encrypted_ops;
mod encrypted_utils;
mod encrypted_losses;
mod network;
mod encrypted_nn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = EncryptedNeuralNetworkU16GPU::create();
    model.add_dense(2, 3);
    let id = String::from("Dense1");
    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    model.print_plain_grad_weights(id.clone());
    model.print_plain_grad_biases(id.clone());
    model.train(
        1,                          
        2,                           
        0.01,                       
        &[vec![1.0, 1.0], vec![2.0, 2.0]],               
        &[vec![2.6, 4.0, 1.3], vec![5.2, 8.0, 2.6]],              
        &[vec![2.0, 8.0]],               
        &[vec![5.2, 9.0, 2.0]],               
    );
    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    //model.print_plain_grad_weights(id.clone());
    //model.print_plain_grad_biases(id.clone());
    // Plain
    /* 
    let mut plain_dense = PlainDenseLayer::new(2, 3);
    plain_dense.train(0.01, 2, 10, &[vec![1.0, 1.0], vec![2.0, 2.0]],  &[vec![2.6, 4.0, 1.3], vec![5.2, 8.0, 2.6]]);
    */
    Ok(())
}