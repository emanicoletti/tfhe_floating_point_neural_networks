use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::encrypted_losses::loss_function::LossFunction;
use crate::encrypted_losses::loss_function::MseLoss;
use crate::encrypted_ops::*;
use crate::encrypted_nn;
use crate::encrypted_nn::EncryptedNeuralNetworkImpl;

use tfhe::prelude::FheTryEncrypt;

use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::{generate_keys, set_server_key, ClientKey, CompressedServerKey, ConfigBuilder, CudaServerKey, FheUint16, FheUint32, FheUint64, FheUint8, ServerKey};

pub trait EncryptedNeuralNetwork{
    fn create();
}
pub struct EncryptedNeuralNetworkU8CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint8>,
}

pub struct EncryptedNeuralNetworkU8GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint8>,
}

pub struct EncryptedNeuralNetworkU16CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint16>,
}

pub struct EncryptedNeuralNetworkU16GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint16>,
}

pub struct EncryptedNeuralNetworkU32CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint32>,
}

pub struct EncryptedNeuralNetworkU32GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint32>,
}

pub struct EncryptedNeuralNetworkU64CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint64>,
}

pub struct EncryptedNeuralNetworkU64GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint64>,
}

impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU16GPU {
    fn create() {
        
    }
}