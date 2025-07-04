use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::{set_server_key};
use tfhe::{ ConfigBuilder, ClientKey, CompressedServerKey, CudaServerKey};

mod encrypted_layers;
mod plain_layers;
mod encrypted_ops;
mod encrypted_utils;
use crate::encrypted_layers::EncryptedDenseLayer;
use crate::plain_layers::PlainDenseLayer;

fn main() {
    
    // 1. Generate keys
    let config = ConfigBuilder::with_custom_parameters(
        PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
    )
    .build();

    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let cuda_server_key = compressed_server_key.decompress_to_gpu();


    set_server_key(cuda_server_key.clone());
}