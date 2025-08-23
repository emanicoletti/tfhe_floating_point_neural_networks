mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;
mod experiment_3;
mod ops_playground;

use crate::experiment_1_2::exp_1_settings::*;
use crate::experiment_1_2::exp_2_settings::*;
use crate::experiment_3::exp_3_settings::*;
use crate::ops_playground::test_ops::ops_playground_info;


fn main() -> Result<(), Box<dyn std::error::Error>> {

    ops_playground_info();
    //experiment_1_fp32(true, false, true, false)?;
    /* 

    let mut plain_model = PlainNeuralNetworkU32::create();
    
    plain_model.add_conv(1, 32, 3, 3, 1, 1);
    plain_model.add_batch_norm(32);
    plain_model.add_relu_activation(32);
    plain_model.add_max_pooling(vec![28, 28], 2, 2, 0);

    plain_model.add_conv(32, 64, 3, 3, 1, 1);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(64);
    plain_model.add_max_pooling(vec![14, 14], 2, 2, 0);

    plain_model.add_conv(64, 128, 3, 3, 1, 1);
    plain_model.add_batch_norm(128);
    plain_model.add_relu_activation(128);

    plain_model.add_dense(6272, 256);
    plain_model.add_relu_activation(256);
    plain_model.add_dense(256, 10);
    
    plain_model.print_plain_weights(String::from("Conv1"));
    plain_model.print_plain_biases(String::from("Conv1"));
    plain_model.print_plain_weights(String::from("Conv5"));
    plain_model.print_plain_biases(String::from("Conv5"));
    plain_model.print_plain_weights(String::from("Conv9"));
    plain_model.print_plain_biases(String::from("Conv9"));
    plain_model.print_plain_weights(String::from("Dense12"));
    plain_model.print_plain_biases(String::from("Dense12"));
    plain_model.print_plain_weights(String::from("Dense14"));
    plain_model.print_plain_biases(String::from("Dense14"));
    

    plain_model.train(
        1,
        64,
        0.01,
        &train_inputs,
        &train_labels,
        vec![640, 1, 28, 28],
        vec![640, 1, 1, 10],
    );

    */
    
    Ok(())
}





