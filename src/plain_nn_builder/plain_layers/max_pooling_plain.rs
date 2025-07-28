pub struct MaxPoolingLayer {
    input_dim: Vec<usize>, // [batch, channels, height, width]
    kernel_size: usize,
    stride: usize,
    padding: usize,
    id: String,
}

impl MaxPoolingLayer {
    pub fn new(id: String, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            input_dim,
            kernel_size,
            stride,
            padding,
            id,
        }
    }

    pub fn forward(&self, input: &Vec<Vec<Vec<Vec<u32>>>>) -> Vec<Vec<Vec<Vec<u32>>>> {
        let (batch, channels, height, width) = (self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]);
        let kernel = self.kernel_size;
        let stride = self.stride;

        let out_height = (height - kernel) / stride + 1;
        let out_width = (width - kernel) / stride + 1;

        let mut output = vec![vec![vec![vec![0u32; out_width]; out_height]; channels]; batch];

        for n in 0..batch {
            for c in 0..channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let mut max_val = 0u32;
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;
                                if ih < height && iw < width {
                                    let val = input[n][c][ih][iw];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                        output[n][c][h][w] = max_val;
                    }
                }
            }
        }

        output
    }

    pub fn backward(&self, input: &Vec<Vec<Vec<Vec<u32>>>>, grad_output: &Vec<Vec<Vec<Vec<u32>>>>) -> Vec<Vec<Vec<Vec<u32>>>> {
        let (batch, channels, height, width) = (self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]);
        let kernel = self.kernel_size;
        let stride = self.stride;

        let out_height = (height - kernel) / stride + 1;
        let out_width = (width - kernel) / stride + 1;

        let mut grad_input = vec![vec![vec![vec![0u32; width]; height]; channels]; batch];

        for n in 0..batch {
            for c in 0..channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let mut max_val = 0u32;
                        let mut max_pos = (0, 0);
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;
                                if ih < height && iw < width {
                                    let val = input[n][c][ih][iw];
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = (ih, iw);
                                    }
                                }
                            }
                        }
                        grad_input[n][c][max_pos.0][max_pos.1] = grad_input[n][c][max_pos.0][max_pos.1].saturating_add(grad_output[n][c][h][w]);
                    }
                }
            }
        }

        grad_input
    }

    pub fn update_parameters(&mut self, _learning_rate: u32) {
        // No parameters to update in max pooling
    }

    pub fn get_weights(&self) -> Vec<u32> {
        vec![] // No weights in max pooling
    }

    pub fn get_biases(&self) -> Vec<u32> {
        vec![] // No biases in max pooling
    }

    pub fn get_grad_weights(&self) -> Vec<u32> {
        vec![] // No grad weights in max pooling
    }

    pub fn get_grad_biases(&self) -> Vec<u32> {
        vec![] // No grad biases in max pooling
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }
}

