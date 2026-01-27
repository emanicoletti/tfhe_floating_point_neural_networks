use crate::plain_nn_builder::plain_nn::{self, PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn vgg_fp32(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool) -> Result<(), Box<dyn std::error::Error>> {
    
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    // Load data from folder experiment_1_2 
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(6)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(8));

    if train_plain_network {

        // Define the architecture
        // --- BLOCK 1 ---
        // Input: 28x28x3
        plain_model.add_conv(3, 64, 3, 3, 1, 0); 
        plain_model.add_batch_norm(64);
        plain_model.add_relu_activation(43264); 

        plain_model.add_conv(64, 64, 3, 3, 1, 0);
        plain_model.add_batch_norm(64);
        plain_model.add_relu_activation(36864); 

        plain_model.add_max_pooling(vec![24, 24], 2, 2); 

        // --- BLOCK 2 ---
        plain_model.add_conv(64, 96, 3, 3, 1, 0);
        plain_model.add_batch_norm(96);
        plain_model.add_relu_activation(9600); 

        plain_model.add_conv(96, 96, 3, 3, 1, 0);
        plain_model.add_batch_norm(96);
        plain_model.add_relu_activation(6144); 

        plain_model.add_max_pooling(vec![8, 8], 2, 2); 

        // --- CLASSIFIER ---
        plain_model.add_dense(1536, 128); 
        plain_model.add_relu_activation(128);
        plain_model.add_dense(128, 7);
        plain_model.add_relu_activation(7);

        // Train and validate the model
        plain_model.train_and_validate(
            15,
            64,
            0.1,
            0.0,
            0.0,
            &train_inputs,
            &train_labels,
            &test_inputs,
            &test_labels,
            vec![8010, 3, 28, 28],
            vec![8010, 1, 1, 7],
            vec![2005, 3, 28, 28],
            vec![2005, 1, 1, 7],
        );

    }

    if test_plain_network {

        // Validate the plain model
        let mut correct = 0;
        let total = test_labels.len();

        for (input, label) in test_inputs.iter().zip(test_labels.iter()) {
            let input_batch = vec![input.clone()]; 
            let input_shape = vec![1, 1, 28, 28];
            let label_shape = vec![1, 1, 1, 10];

            let prediction = plain_model.inference(&input_batch, input_shape, label_shape);

            let predicted_class = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let actual_class = label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_class == actual_class {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / total as f32;
        println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    }

    Ok(())
}

pub fn sc_mnist_exp_fp32(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool) -> Result<(), Box<dyn std::error::Error>> {
    
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    // Load data from folder experiment_1_2 
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(6)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(7));

    if train_plain_network {

        // Define the architecture
        plain_model.add_conv(3, 64, 3, 3, 1, 0);
        plain_model.add_batch_norm(64);
        plain_model.add_relu_activation(43264);
        plain_model.add_avg_pooling(vec![26, 26], 2, 2);
        plain_model.add_conv(64, 96, 3, 3, 1, 0);
        plain_model.add_batch_norm(96);
        plain_model.add_relu_activation(11616);
        plain_model.add_avg_pooling(vec![11, 11], 2, 2);
        plain_model.add_dense(2400, 128);
        plain_model.add_relu_activation(128);
        plain_model.add_dense(128, 7);
        plain_model.add_relu_activation(7);

        // Train and validate the model
        plain_model.train_and_validate(
            15,
            60,
            0.05,
            0.0,
            0.0,
            &train_inputs,
            &train_labels,
            &test_inputs,
            &test_labels,
            vec![8010, 3, 28, 28],
            vec![8010, 1, 1, 7],
            vec![2005, 3, 28, 28],
            vec![2005, 1, 1, 7],
        );

    }

    if test_plain_network {

        // Validate the plain model
        let mut correct = 0;
        let total = test_labels.len();

        for (input, label) in test_inputs.iter().zip(test_labels.iter()) {
            let input_batch = vec![input.clone()]; 
            let input_shape = vec![1, 1, 28, 28];
            let label_shape = vec![1, 1, 1, 10];

            let prediction = plain_model.inference(&input_batch, input_shape, label_shape);

            let predicted_class = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let actual_class = label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_class == actual_class {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / total as f32;
        println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    }

    Ok(())
}