use crate::plain_nn_builder::plain_layers::PlainLayer;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;


pub struct GlobalAveragePooling<T> 
where T: PlainElement 
{
    id: String,
    // Nessun peso da imparare!
    _marker: std::marker::PhantomData<T>,
}

impl<T> GlobalAveragePooling<T> 
where T: PlainElement 
{
    pub fn new(id: String) -> Self {
        Self { 
            id,
            _marker: std::marker::PhantomData 
        }
    }
}

impl<T> PlainLayer<T> for GlobalAveragePooling<T> 
where T: PlainElement + PlainMul + PlainAdd + PlainDiv + PlainValueType + Default 
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let shape = input.clone().shape; // [Batch, Channel, Height, Width]
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        
        let spatial_dim = height * width;
        let scale_factor = T::from_f32(1.0 / (spatial_dim as f32)); // Assumiamo un metodo di conversione

        let mut output_data = Vec::with_capacity(batch_size * channels);

        // L'input data è piatto: [Batch 0 (Ch0... Ch1...), Batch 1...]
        // Ogni "blocco" di dati che ci interessa è lungo 'spatial_dim'
        
        // Iteriamo sui blocchi di dimensione 'spatial_dim'
        // chunk è una slice che rappresenta una singola feature map (H x W)
        for feature_map in input.data.chunks(spatial_dim) {
            let sum: T = feature_map.iter().fold(T::default(), |acc, x| acc.add(x.clone()));
            let avg = sum.mul(scale_factor.clone());
            output_data.push(avg);
        }

        // L'output ora è [Batch, Channel, 1, 1]
        // Questo è compatibile con un Dense layer che si aspetta [Batch, InputSize]
        // dato che la dimensione dei dati sottostanti è identica.
        PlainTensor::new(output_data, vec![batch_size, channels, 1, 1])
    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>, 
        grad_output: &PlainTensor<T>, 
    ) -> PlainTensor<T> {
        // grad_output shape: [Batch, Channels, 1, 1]
        // input shape: [Batch, Channels, H, W]
        // Vogliamo espandere ogni valore di grad_output ripetendolo H*W volte e dividendolo.
        
        let shape = input.clone().shape;
        let spatial_dim = shape[2] * shape[3];
        let scale_factor = T::from_f32(1.0 / (spatial_dim as f32));

        let mut grad_input_data = Vec::with_capacity(input.data.len());

        // Per ogni gradiente singolo (che corrisponde a un canale)
        for grad_val in grad_output.data.iter() {
            // Calcoliamo il valore da distribuire: grad / (H*W)
            let distributed_grad = grad_val.clone().mul(scale_factor.clone());
            
            // Lo ripetiamo per tutti i pixel di quella feature map
            for _ in 0..spatial_dim {
                grad_input_data.push(distributed_grad.clone());
            }
        }

        PlainTensor::new(grad_input_data, shape.clone())
    }

    fn update_parameters(&mut self, _learning_rate: T) {
        // GAP non ha parametri da aggiornare
    }
    
    // ... Altri metodi boilerplate (inference è uguale a forward, get_weights panic o empty, etc.)
    fn get_id(&self) -> String { self.id.clone() }
    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> { self.forward(input) }
    fn approximate_inference(&mut self, _input: &PlainTensor<T>) -> PlainTensor<T> { todo!() }
    fn get_weights(&self) -> PlainTensor<T> { PlainTensor::new(vec![], vec![]) } 
    fn get_biases(&self) -> PlainTensor<T> { PlainTensor::new(vec![], vec![]) }
    fn get_grad_weights(&self) -> PlainTensor<T> { PlainTensor::new(vec![], vec![]) }
    fn get_grad_biases(&self) -> PlainTensor<T> { PlainTensor::new(vec![], vec![]) }
}