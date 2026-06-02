use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;
use crate::plain_nn_builder::plain_layers::residual_block_plain::ResidualBlock;

#[allow(dead_code)]
pub fn resnet20_fp32() -> Result<(), Box<dyn std::error::Error>> {

    // Load data from folder
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(8)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(9));

    let mut residual_block_1 = ResidualBlock::<u32>::new(
        "res_block_1".to_string(),
        Some(9)
    );

    let mut residual_block_2 = ResidualBlock::<u32>::new(
        "res_block_2".to_string(),
        Some(9) 
    );

    let mut residual_block_3 = ResidualBlock::<u32>::new(
        "res_block_3".to_string(),
        Some(9)
    );

    let mut residual_block_4 = ResidualBlock::<u32>::new(
        "res_block_4".to_string(),
        Some(9)
    );

    let mut residual_block_5 = ResidualBlock::<u32>::new(
        "res_block_5".to_string(),
        Some(9)
    );

    let mut residual_block_6 = ResidualBlock::<u32>::new(
        "res_block_6".to_string(),
        Some(9)
    );

    let mut residual_block_7 = ResidualBlock::<u32>::new(
        "res_block_7".to_string(),
        Some(9)
    );

    let mut residual_block_8 = ResidualBlock::<u32>::new(
        "res_block_8".to_string(),
        Some(9)
    );

    let mut residual_block_9 = ResidualBlock::<u32>::new(
        "res_block_9".to_string(),
        Some(9)
    );


        plain_model.add_conv(3, 16, 3, 3, 1, 1);
        plain_model.add_batch_norm(16);

        // First Stage
        residual_block_1.add_conv(false, 16, 16, 3, 3, 1, 1, "Rb1_conv1".to_string());
        residual_block_1.add_batch_norm(false, 16, "Rb1_bn1".to_string());
        residual_block_1.add_relu_activation(false, 16*32*32, "Rb1_relu1".to_string());
        
        residual_block_1.add_conv(false, 16, 16, 3, 3, 1, 1, "Rb1_conv2".to_string());
        residual_block_1.add_batch_norm(false, 16, "Rb1_bn2".to_string());

        plain_model.add_residual_block(residual_block_1);
        plain_model.add_relu_activation(16*32*32);

        // Second Stage
        residual_block_2.add_conv(false, 16, 16, 3, 3, 1, 1, "Rb2_conv1".to_string());
        residual_block_2.add_batch_norm(false, 16, "Rb2_bn1".to_string());
        residual_block_2.add_relu_activation(false, 16*32*32, "Rb2_relu1".to_string());
        
        residual_block_2.add_conv(false, 16, 16, 3, 3, 1, 1, "Rb2_conv2".to_string());
        residual_block_2.add_batch_norm(false, 16, "Rb2_bn2".to_string());

        plain_model.add_residual_block(residual_block_2);
        plain_model.add_relu_activation(16*32*32);

        // Third Stage
        residual_block_3.add_conv(false, 16, 16, 3, 3, 1, 1, "Rb3_conv1".to_string());
        residual_block_3.add_batch_norm(false, 16, "Rb3_bn1".to_string());
        residual_block_3.add_relu_activation(false, 16*32*32, "Rb3_relu1".to_string());
        
        residual_block_3.add_conv(false, 16, 16, 3, 3, 1, 1, "Rb3_conv2".to_string());
        residual_block_3.add_batch_norm(false, 16, "Rb3_bn2".to_string());

        plain_model.add_residual_block(residual_block_3);
        plain_model.add_relu_activation(16*32*32);

        // Fourth Stage
        residual_block_4.add_conv(false, 16, 32, 3, 3, 2, 1, "Rb4_conv1".to_string());
        residual_block_4.add_batch_norm(false, 32, "Rb4_bn1".to_string());
        residual_block_4.add_relu_activation(false, 32*16*16, "Rb4_relu1".to_string());
        
        residual_block_4.add_conv(false, 32, 32, 3, 3, 1, 1, "Rb4_conv2".to_string());
        residual_block_4.add_batch_norm(false, 32, "Rb4_bn2".to_string());

        residual_block_4.add_conv(true, 16, 32, 1, 1, 2, 0, "Rb4_skip_conv".to_string());
        residual_block_4.add_batch_norm(true, 32, "Rb4_skip_bn".to_string());

        plain_model.add_residual_block(residual_block_4);
        plain_model.add_relu_activation(32*16*16);

        // Fifth Stage
        residual_block_5.add_conv(false, 32, 32, 3, 3, 1, 1, "Rb5_conv1".to_string());
        residual_block_5.add_batch_norm(false, 32, "Rb5_bn1".to_string());
        residual_block_5.add_relu_activation(false, 32*16*16, "Rb5_relu1".to_string());
        
        residual_block_5.add_conv(false, 32, 32, 3, 3, 1, 1, "Rb5_conv2".to_string());
        residual_block_5.add_batch_norm(false, 32, "Rb5_bn2".to_string());

        plain_model.add_residual_block(residual_block_5);
        plain_model.add_relu_activation(32*16*16);

        // Sixth Stage
        residual_block_6.add_conv(false, 32, 32, 3, 3, 1, 1, "Rb6_conv1".to_string());
        residual_block_6.add_batch_norm(false, 32, "Rb6_bn1".to_string());
        residual_block_6.add_relu_activation(false, 32*16*16, "Rb6_relu1".to_string());
        
        residual_block_6.add_conv(false, 32, 32, 3, 3, 1, 1, "Rb6_conv2".to_string());
        residual_block_6.add_batch_norm(false, 32, "Rb6_bn2".to_string());

        plain_model.add_residual_block(residual_block_6);
        plain_model.add_relu_activation(32*16*16);

        // Seventh Stage
        residual_block_7.add_conv(false, 32, 64, 3, 3, 2, 1, "Rb7_conv1".to_string());
        residual_block_7.add_batch_norm(false, 64, "Rb7_bn1".to_string());
        residual_block_7.add_relu_activation(false, 64*8*8, "Rb7_relu1".to_string());
        
        residual_block_7.add_conv(false, 64, 64, 3, 3, 1, 1, "Rb7_conv2".to_string());
        residual_block_7.add_batch_norm(false, 64, "Rb7_bn2".to_string());

        residual_block_7.add_conv(true, 32, 64, 1, 1, 2, 0, "Rb7_skip_conv".to_string());
        residual_block_7.add_batch_norm(true, 64, "Rb7_skip_bn".to_string());

        plain_model.add_residual_block(residual_block_7);
        plain_model.add_relu_activation(64*8*8);

        // Eighth Stage
        residual_block_8.add_conv(false, 64, 64, 3, 3, 1, 1, "Rb8_conv1".to_string());
        residual_block_8.add_batch_norm(false, 64, "Rb8_bn1".to_string());
        residual_block_8.add_relu_activation(false, 64*8*8, "Rb8_relu1".to_string());
        
        residual_block_8.add_conv(false, 64, 64, 3, 3, 1, 1, "Rb8_conv2".to_string());
        residual_block_8.add_batch_norm(false, 64, "Rb8_bn2".to_string());

        plain_model.add_residual_block(residual_block_8);
        plain_model.add_relu_activation(64*8*8);

        // Ninth Stage
        residual_block_9.add_conv(false, 64, 64, 3, 3, 1, 1, "Rb9_conv1".to_string());
        residual_block_9.add_batch_norm(false, 64, "Rb9_bn1".to_string());
        residual_block_9.add_relu_activation(false, 64*8*8, "Rb9_relu1".to_string());
        
        residual_block_9.add_conv(false, 64, 64, 3, 3, 1, 1, "Rb9_conv2".to_string());
        residual_block_9.add_batch_norm(false, 64, "Rb9_bn2".to_string());

        plain_model.add_residual_block(residual_block_9);
        plain_model.add_relu_activation(64*8*8);

        // Global Average Pooling
        plain_model.add_global_avg_pooling();
        
        // Final Classifier
        plain_model.add_dense(64, 10);

        plain_model.train_and_validate(
            25, 
            64, 
            0.1, 
            0.0001,
            0.9,
            &train_inputs, 
            &train_labels, 
            &test_inputs, 
            &test_labels, 
            vec![100000, 3, 32, 32],
            vec![100000, 1, 1, 10],
            vec![10000, 3, 32, 32],
            vec![10000, 1, 1, 10],
        );

    Ok(())

    }
