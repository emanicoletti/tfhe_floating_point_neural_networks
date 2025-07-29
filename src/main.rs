use tfhe::boolean::backward_compatibility::server_key;
use tfhe::shortint::parameters::*;
use tfhe::shortint::parameters::v1_2::*;
use tfhe::{set_server_key};
use tfhe::{ConfigBuilder, ClientKey, generate_keys, CompressedServerKey, CudaServerKey, FheUint8, FheUint16, FheUint32, FheUint64};
use std::time::Instant;
use rand::Rng;
use half::f16;
use tfhe::prelude::*;

use crate::tfhe_nn_builder::encrypted_nn::{EncryptedNeuralNetwork, EncryptedNeuralNetworkU32GPU};
use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};

mod tfhe_nn_builder;
mod plain_nn_builder;

use rayon::ThreadPoolBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {

      
    let train_inputs = &[
        vec![0.000778, 0.061168, 0.247278, 0.035494, 0.030891, 0.331023, 0.412921, 0.162872, 0.102673, 0.310408, 0.255427, 0.106880, 0.046329, 0.263268, 0.100277, 0.004832],
        //vec![0.000002, 0.003925, 0.106786, 0.033819, 0.000453, 0.075774, 0.348921, 0.040959, 0.011279, 0.322118, 0.162577, 0.002102, 0.018774, 0.234882, 0.026033, 0.000026],
        //vec![0.001184, 0.055320, 0.134542, 0.013901, 0.011918, 0.289323, 0.455587, 0.062753, 0.099947, 0.424732, 0.436236, 0.146384, 0.058734, 0.134263, 0.032525, 0.019126],
        //vec![0.000843, 0.098897, 0.082685, 0.000508, 0.001346, 0.203403, 0.253636, 0.002771, 0.000555, 0.154521, 0.357554, 0.008308, 0.000074, 0.040729, 0.181102, 0.006945],
        //vec![0.000191, 0.036873, 0.039497, 0.000234, 0.000667, 0.137136, 0.166216, 0.001220, 0.000372, 0.109462, 0.199726, 0.002248, 0.000098, 0.044278, 0.106672, 0.001425]
    ];

    /* 
    let train_inputs = &[
        vec![0.000778, 0.061168, 0.247278, 0.035494],
        //vec![0.000002, 0.003925, 0.106786, 0.033819, 0.000453, 0.075774, 0.348921, 0.040959, 0.011279, 0.322118, 0.162577, 0.002102, 0.018774, 0.234882, 0.026033, 0.000026],
        //vec![0.001184, 0.055320, 0.134542, 0.013901, 0.011918, 0.289323, 0.455587, 0.062753, 0.099947, 0.424732, 0.436236, 0.146384, 0.058734, 0.134263, 0.032525, 0.019126],
        //vec![0.000843, 0.098897, 0.082685, 0.000508, 0.001346, 0.203403, 0.253636, 0.002771, 0.000555, 0.154521, 0.357554, 0.008308, 0.000074, 0.040729, 0.181102, 0.006945],
        //vec![0.000191, 0.036873, 0.039497, 0.000234, 0.000667, 0.137136, 0.166216, 0.001220, 0.000372, 0.109462, 0.199726, 0.002248, 0.000098, 0.044278, 0.106672, 0.001425]
    ];
    */
    /* 
    let train_inputs = &[
        vec![0.000000, 0.000000, 0.000000, 0.000375, 0.017839, 0.017480, 0.000243, 0.000000, 0.000000, 0.000000, 0.000781, 0.068231, 0.563594, 0.504285, 0.028150, 0.000031, 0.000000, 0.000255, 0.068295, 0.626798, 0.766264, 0.605544, 0.210780, 0.000679, 0.000006, 0.018010, 0.473337, 0.405960, 0.081385, 0.289549, 0.447691, 0.004281, 0.000151, 0.120197, 0.566282, 0.026495, 0.014719, 0.361949, 0.345463, 0.002837, 0.000210, 0.146584, 0.533963, 0.163561, 0.410760, 0.361678, 0.030780, 0.000074, 0.000085, 0.073841, 0.640273, 0.641265, 0.253348, 0.022373, 0.000067, 0.000000, 0.000001, 0.001185, 0.026416, 0.027628, 0.002206, 0.000020, 0.000000, 0.000000]
    ];
    */

    let train_labels = &[
        vec![1.000000, 0.000000, 0.000000],
        //vec![0.000000, 1.000000, 0.000000],
        //vec![0.000000, 0.000000, 1.000000],
        //vec![0.000000, 1.000000, 0.000000],
        //vec![0.000000, 1.000000, 0.000000]
    ];
    
    
    let mut model = EncryptedNeuralNetworkU32GPU::create();
    model.add_max_pooling(vec![4, 4], 2, 2, 0);
    model.add_dense(4, 3);
    model.add_tanh_activation(3);
    /*
    model.add_dense(4, 2);
    model.add_tanh_activation(2);
    model.add_dense(2, 3);
    model.add_tanh_activation(3);
    */
    //model.add_dense(10, 3);
    let id = String::from("Dense2");
    //let id1 = String::from("Dense2");
    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    //model.print_plain_grad_weights(id.clone());
    //model.print_plain_grad_biases(id.clone());
    
    
    model.train(
        1,                          
        1,                           
        0.01,                       
        train_inputs,              
        train_labels,             
        vec![1, 1, 4, 4],
        vec![1, 1, 1, 3]          
    );


    
    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    //model.print_plain_weights(id1.clone());
    //model.print_plain_biases(id1.clone());
    
    
    let mut plain_model = PlainNeuralNetworkU32::create();
    plain_model.add_max_pooling(vec![4, 4], 2, 2, 0);
    plain_model.add_dense(4, 3);
    plain_model.add_tanh_activation(3);

    let id = String::from("Dense2");
    //let id1 = String::from("Dense2");
    plain_model.print_plain_weights(id.clone());
    plain_model.print_plain_biases(id.clone());


    plain_model.train(
        1,                          
        1,                           
        0.01,                       
        train_inputs,              
        train_labels,             
        vec![1, 1, 4, 4],
        vec![1, 1, 1, 3]          
    );

    Ok(())
}