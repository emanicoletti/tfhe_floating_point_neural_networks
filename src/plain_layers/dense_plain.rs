use half::f16;

pub struct PlainDenseLayer {
    weights: Vec<Vec<f16>>, // [output_size][input_size]
    biases: Vec<f16>,       // [output_size]
}

impl PlainDenseLayer {
    pub fn new(weights: Vec<Vec<f16>>, biases: Vec<f16>) -> Self {
        assert_eq!(weights.len(), biases.len(), "Weights and biases size mismatch");
        Self { weights, biases }
    }

    pub fn forward(&self, input: &[f16]) -> Vec<f16> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(weight_row, bias)| {
                let sum: f32 = weight_row.iter().zip(input.iter())
                    .map(|(w, x)| w.to_f32() * x.to_f32())
                    .sum();
                f16::from_f32(sum + bias.to_f32())
            })
            .collect()
    }

    pub fn backward(
        &mut self,
        input: &[f16],
        target: &[f16],
        output: &[f16],
        learning_rate: f16,
    ) {
        let lr = learning_rate.to_f32();

        for ((weight_row, bias), (y_hat, y_true)) in self.weights.iter_mut()
            .zip(self.biases.iter_mut())
            .zip(output.iter().zip(target.iter())) {

            let error = y_hat.to_f32() - y_true.to_f32();
            println!("Error: {}", error);

            // Update weights
            for (w_i, x_i) in weight_row.iter_mut().zip(input.iter()) {
                let grad = error * x_i.to_f32();
                let updated = w_i.to_f32() - lr * grad;
                println!("Weight update: {}", - lr * grad);
                *w_i = f16::from_f32(updated);
            }

            

            // Update bias
            let updated_bias = bias.to_f32() - lr * error;
            *bias = f16::from_f32(updated_bias);
        }
    }

    pub fn print_learned_parameters(&self) {
        println!("\nLearned Weights:");
        for (i, row) in self.weights.iter().enumerate() {
            let decoded: Vec<f32> = row.iter().map(|w| w.to_f32()).collect();
            println!("Neuron {}: {:?}", i, decoded);
        }

        println!("\nLearned Biases:");
        let decoded_biases: Vec<f32> = self.biases.iter().map(|b| b.to_f32()).collect();
        println!("{:?}", decoded_biases);
    }
}