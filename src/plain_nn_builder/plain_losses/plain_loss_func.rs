use crate::plain_nn_builder::{plain_ops::{PlainAdd, PlainDiv, PlainMul, PlainSub}, plain_utils::{PlainElement, PlainTensor, PlainValueType}};

pub trait PlainLossFunction<T>
where 
    T: PlainElement
{
    fn compute_loss(
        &self,
        predicted: &PlainTensor<T>,
        target: &PlainTensor<T>,
    ) -> PlainTensor<T>;

    fn gradient(
        &self,
        predicted: &PlainTensor<T>,
        target: &PlainTensor<T>,
    ) -> PlainTensor<T>;

}

pub struct MseLoss;

impl<T> PlainLossFunction<T> for MseLoss
where
    T: Default + PlainSub + PlainMul + PlainAdd + PlainDiv + Sync + Send + Clone + PlainElement + PlainValueType,
{
    fn compute_loss(
        &self,
        predicted: &PlainTensor<T>,
        target: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let mut sum = T::default();

        for (p, t) in predicted.data.iter().zip(&target.data) {
            let diff = p.clone().sub(t.clone());
            let squared = diff.clone().mul(diff.clone());
            sum = sum.clone().add(squared.clone());
        }

        let n = predicted.data.len() as f32;

        PlainTensor {
            data: vec![sum.div(T::from_f32(n))],
            shape: [1,1].to_vec(),
        }
    }

    fn gradient(
        &self,
        predicted: &PlainTensor<T>,
        target: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let n = predicted.data.len() as f32;

        let two = 2.0 as f32;
        let two_bits = two.to_bits();
    
        let mut grad_data = Vec::with_capacity(n as usize);
    
        for (p, t) in predicted.data.iter().zip(&target.data) {
            let diff = p.clone().sub(t.clone());
            //println!("Diff: {:?} - {:?} = {:?}", p.clone().to_f32(), t.clone().to_f32(), diff.clone().to_f32());
            let double_diff = diff.clone().mul(T::from_f32(2.0 as f32));
            //println!("Double diff: {:?}", double_diff.clone().to_f32());
            let grad = double_diff.clone().div(T::from_f32(n));
            //println!("Grad: {:?}", grad.clone().to_f32());

            grad_data.push(grad);
        }
    
        PlainTensor {
            data: grad_data,
            shape: predicted.shape.clone(),
        }
    }

}