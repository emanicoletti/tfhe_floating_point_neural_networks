
use crate::plain_nn_builder::{plain_layers::{PlainLayer, PlainDenseLayer, PlainConv2DLayer, PlainMaxPoolingLayer, PlainAvgPoolingLayer, PlainBatchNormLayer, ResidualBlock, GlobalAveragePooling}, plain_losses::PlainLossFunction, plain_ops::*, plain_utils::{PlainElement, PlainValueType, PlainTensor}, plain_activations::{PlainTanhActivation, PlainReLUActivation}};
use rayon::prelude::*;

pub struct PlainNeuralNetworkImpl<T: PlainElement> {
    pub layers: Vec<Box<dyn PlainLayer<T>>>,
    pub loss: Box<dyn PlainLossFunction<T>>,
}

impl<T> PlainNeuralNetworkImpl<T>
where 
    T: PlainAdd
    + PlainMul
    + PlainDiv
    + PlainSub
    + PlainTanh
    + PlainReLU
    + PlainBackwardReLU
    + PlainSqrt
    + PlainDivInf
    + PlainMulInf
    + PlainElement
    + PlainValueType
    + Clone
    + Copy
    + Ord
    + Default
    + 'static
{

    pub fn add_dense(&mut self, weights: PlainTensor<T>, biases: PlainTensor<T>, grad_weights: PlainTensor<T>, grad_biases: PlainTensor<T>) {
        let id = format!("Dense{}", self.layers.len() + 1);
        let dense_layer = PlainDenseLayer {
            id: id, 
            weights: weights,
            biases: biases,
            grad_weights: Some(grad_weights),
            grad_biases: Some(grad_biases),
            velocity_weights: None,
            velocity_biases: None,
        };
        self.layers.push(Box::new(dense_layer));
    }

    pub fn add_tanh_activation(&mut self, derivatives: PlainTensor<T>) {
        let id = format!("Tanh{}", self.layers.len() + 1);
        let tanh_layer = PlainTanhActivation{
            id: id,
            derivatives: derivatives,
        };
        self.layers.push(Box::new(tanh_layer));
    }

    pub fn add_relu_activation(&mut self, derivatives: PlainTensor<T>) {
        let id = format!("ReLU{}", self.layers.len() + 1);
        let relu_layer = PlainReLUActivation::new(id, derivatives);
        self.layers.push(Box::new(relu_layer));
    }

    pub fn add_max_pooling(
        &mut self,
        input_dim: Vec<usize>,
        kernel_size: usize,
        stride: usize,
    ) {
        let id = format!("MaxPooling{}", self.layers.len() + 1);
        let max_pooling_layer = PlainMaxPoolingLayer::new(id, input_dim, kernel_size, stride);
        self.layers.push(Box::new(max_pooling_layer));
    }

    pub fn add_avg_pooling(
        &mut self,
        input_dim: Vec<usize>,
        kernel_size: usize,
        stride: usize,
    ) {
        let id = format!("AvgPooling{}", self.layers.len() + 1);
        let avg_pooling_layer = PlainAvgPoolingLayer::new(id, input_dim, kernel_size, stride);
        self.layers.push(Box::new(avg_pooling_layer));
    }

    pub fn add_conv(&mut self, weights: PlainTensor<T>, biases: PlainTensor<T>, grad_weights: PlainTensor<T>, grad_biases: PlainTensor<T>, stride: usize, padding: usize) {
        let id = format!("Conv{}", self.layers.len() + 1);
        let conv_layer = PlainConv2DLayer {
            id: id,
            weights: weights,
            biases: biases,
            grad_weights: Some(grad_weights),
            grad_biases: Some(grad_biases),
            stride,
            padding,
            velocity_weights: None,
            velocity_biases: None,
        };
        self.layers.push(Box::new(conv_layer));
    }

    pub fn add_batch_norm(
        &mut self,
        x_hat: PlainTensor<T>,
        mean: PlainTensor<T>,
        variance: PlainTensor<T>,
        gamma: PlainTensor<T>,
        beta: PlainTensor<T>,
    ) {
        let id = format!("BatchNorm{}", self.layers.len() + 1);
        let batch_norm_layer = PlainBatchNormLayer::new(id, x_hat, mean, variance, gamma, beta);
        self.layers.push(Box::new(batch_norm_layer));
    }

    pub fn add_residual_block(
        &mut self,
        residual_block: ResidualBlock<T>,
    ) {
        self.layers.push(Box::new(residual_block));
    }

    pub fn add_global_avg_pooling(&mut self) {
        let id = format!("GlobalAvgPooling{}", self.layers.len() + 1);
        let global_avg_pooling_layer = GlobalAveragePooling::new(id);
        self.layers.push(Box::new(global_avg_pooling_layer));
    }

    pub fn train( 
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: T,
        weight_decay: T,
        momentum: T,
        train_inputs: PlainTensor<T>,
        train_labels: PlainTensor<T>,
    ) 
    {   
        for epoch in 0..epochs{
            println!("Epoch {}/{}", epoch + 1, epochs);
            let mut i_batch = 1;
            for (input_batch, label_batch) in self.iter_batches(&train_inputs, &train_labels, batch_size){
                let mut activations = vec![input_batch.clone()];
                for layer in &mut self.layers {
                    let output = layer.forward(activations.last().unwrap());
                    activations.push(output.clone());
                }
                
                let prediction = activations.last().unwrap();
                let loss_val = self.loss.compute_loss(&prediction, &label_batch);
                //println!("Batch {:?} Loss: {:<6} ", i_batch, loss_val.data[0].to_f32());
                let mut grad = self.loss.gradient(&prediction, &label_batch);
                //grad.print_tensor();
                for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                    let input_to_layer = &activations[activations.len() - 2 - i];
                    grad = layer.backward(input_to_layer, &grad);
                }
                for layer in &mut self.layers{
                    layer.update_parameters(learning_rate.clone(), weight_decay.clone(), momentum.clone());
                }
                i_batch += 1;
            }
        }
    }

    pub fn inference(
        &mut self,
        input: &PlainTensor<T>
    ) -> PlainTensor<T>{
        let mut activations = vec![input.clone()];
        for layer in &mut self.layers {
            let output = layer.inference(activations.last().unwrap());
            activations.push(output.clone());
        }
        let prediction = PlainTensor{
            data: activations.last().unwrap().data.clone(),
            shape: activations.last().unwrap().shape.clone(),
        };

        prediction
    }

    pub fn train_and_validate(
        &mut self,
        epochs: usize,
        batch_size: usize,
        mut learning_rate: T,
        weight_decay: T,
        momentum: T,
        train_inputs: PlainTensor<T>,
        train_labels: PlainTensor<T>,
        val_inputs: PlainTensor<T>,
        val_labels: PlainTensor<T>,
    )
    {   
        for epoch in 0..epochs{
            println!("Epoch {}/{}", epoch + 1, epochs);
            let mut i_batch = 1;
            for (input_batch, label_batch) in self.iter_batches(&train_inputs, &train_labels, batch_size){
                //let start = std::time::Instant::now();
                let mut activations = vec![input_batch.clone()];
                for layer in &mut self.layers {
                    let output = layer.forward(activations.last().unwrap());
                    activations.push(output.clone());
                }
                let prediction = activations.last().unwrap();
                //prediction.print_tensor();
                let loss_val = self.loss.compute_loss(&prediction, &label_batch);
                println!("Batch {:?} Loss: {:<6} ", i_batch, loss_val.data[0].to_f32());
                let mut grad = self.loss.gradient(&prediction, &label_batch);
                for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                    let input_to_layer = &activations[activations.len() - 2 - i];
                    grad = layer.backward(input_to_layer, &grad);
                }
                if epoch >= 15 {
                    learning_rate = T::from_f32(0.01);
                }
                self.layers.par_iter_mut().for_each(|layer| {
                    layer.update_parameters(learning_rate.clone(), weight_decay.clone(), momentum.clone());
                });
                i_batch += 1;
                //let duration = start.elapsed();
                //println!("Time elapsed in batch {} is: {:?}", i_batch, duration);
            }

            let mut correct = 0;
            let total = val_labels.shape[0];

            for (input_batch, label_batch) in self.iter_batches(&val_inputs, &val_labels, 1){
                let prediction = self.inference(&input_batch);
                let mut prediction_f32: Vec<f32> = vec![0.0; label_batch.shape[3]];
                for i in 0..prediction.data.len() {
                    prediction_f32[i] = prediction.data[i].to_f32();
                }

                let predicted_index = prediction_f32
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

                let mut one_hot: Vec<f32> = vec![0.0; prediction_f32.len()];
                one_hot[predicted_index] = 1.0;

                let predicted_class = prediction_f32
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                let label: Vec<f32> = label_batch.data.iter().map(|x| x.to_f32()).collect();

                let actual_class = label
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                //println!("Predicted: {}, Actual: {}", predicted_class, actual_class);

                if predicted_class == actual_class {
                    correct += 1;
                }
            }
            println!("Epoch {:?} Validation Accuracy: {:.2}%", epoch+1, (correct as f32 / total as f32) * 100.0);
        
        }
    }


    fn iter_batches(
        &self,
        inputs: &PlainTensor<T>,
        labels: &PlainTensor<T>,
        batch_size: usize,
    ) -> Vec<(PlainTensor<T>, PlainTensor<T>)> {
        assert!(
            inputs.shape.len() == 4,
            "Expected 4D input tensor, got shape {:?}",
            inputs.shape
        );
        assert!(
            labels.shape.len() == 4 || labels.shape.len() == 2,
            "Expected 2D or 4D label tensor, got shape {:?}",
            labels.shape
        );
    
        let num_samples = inputs.shape[0];
        let input_sample_size = inputs.shape[1] * inputs.shape[2] * inputs.shape[3];
        let label_sample_size: usize = labels.shape.iter().skip(1).product();
    
        let mut batches = Vec::new();
        let mut start = 0;

        while start < num_samples {
            let end = usize::min(start + batch_size, num_samples);
            let input_start = start * input_sample_size;
            let input_end = end * input_sample_size;

            let input_batch_data = inputs.data[input_start..input_end].to_vec();
            let input_batch_shape = vec![
                end - start,
                inputs.shape[1],
                inputs.shape[2],
                inputs.shape[3],
            ];

            let input_batch = PlainTensor {
                data: input_batch_data,
                shape: input_batch_shape,
            };
    
            let label_start = start * label_sample_size;
            let label_end = end * label_sample_size;
            let label_batch_data = labels.data[label_start..label_end].to_vec();
            let mut label_batch_shape = labels.shape.clone();
            label_batch_shape[0] = end - start;
            let label_batch = PlainTensor {
                data: label_batch_data,
                shape: label_batch_shape,
            };
    
            batches.push((input_batch, label_batch));
            start = end;
        }
    
        batches
    }

    fn linear_one_cycle(&self, current_step: usize, max_lr: f32, total_steps: usize, pct_start: f32) -> f32 {
        // 1. Define Constants (Standard OneCycle defaults)
        let div_factor = 25.0;
        let final_div_factor = 10000.0;

        // 2. Calculate Boundaries
        let start_lr = max_lr / div_factor;
        let min_lr = start_lr / final_div_factor;
        
        // Cast strict types to float for calculation
        let current_step_f = current_step as f32;
        let total_steps_f = total_steps as f32;
        let warmup_steps = total_steps_f * pct_start;

        // 3. Phase 1: Warm-up (Linear Increase)
        if current_step_f <= warmup_steps {
            // Progress goes from 0.0 to 1.0
            let progress = current_step_f / warmup_steps;
            
            // Formula: start + (diff * progress)
            return start_lr + (max_lr - start_lr) * progress;
        } 
        // 4. Phase 2: Cool-down (Linear Decrease)
        else {
            let cooldown_steps = total_steps_f - warmup_steps;
            let steps_into_cooldown = current_step_f - warmup_steps;
            
            // Progress goes from 0.0 to 1.0 (Careful: clamp it to 1.0 to avoid going negative)
            let progress = (steps_into_cooldown / cooldown_steps).min(1.0);
            
            // Formula: max - (diff * progress)
            return max_lr - (max_lr - min_lr) * progress;
        }
}

}
