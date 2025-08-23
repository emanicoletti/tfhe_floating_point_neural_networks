// Ops Playground List:
// [add]: Addition - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Slow
// [same_sign_add]: Same Sign Addition - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Slow
// [sub]: Subtraction - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Medium
// [lmul]: Lmul Multiplication - Approximate - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Fast
// [ldiv]: Lmul Division - Approximate - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Fast
// [lmul_tanh]: PLA Tanh based on Lmul - Approximate - Available fp formats: (FP16, FP32) - Execution: Very Slow
// [pam_mul]: PAM Multiplication - Approximate - Available fp formats: (FP16, FP32) - Execution: Fast
// [pam_div]: PAM Division - Approximate - Available fp formats: (FP16, FP32) - Execution: Fast
// [pam_tanh]: PLA Tanh based on PAM - Approximate - Available fp formats: (FP16, FP32) - Execution: Very Slow
// [relu]: ReLU - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Very Fast
// [sqrt]: Square Root - Exact - Available fp formats: (FP16, FP32) - Execution: Very Slow
// [log2]: Base-2 Logarithm - Exact - Available fp formats: (FP16, FP32) - Execution: Extremely Slow


pub fn test_encrypted_ops(ops: &str, fp_size: usize, gpu: bool, num_ops: usize, min_range: f32, max_range: f32) -> Result<(), Box<dyn std::error::Error>> {
    match ops {
        "add" => unimplemented!(),
        "same_sign_add" => unimplemented!(),
        "sub" => unimplemented!(),
        "lmul" => unimplemented!(),
        "ldiv" => unimplemented!(),
        "lmul_tanh" => unimplemented!(),
        "pam_mul" => unimplemented!(),
        "pam_div" => unimplemented!(),
        "pam_tanh" => unimplemented!(),
        "relu" => unimplemented!(),
        "sqrt" => unimplemented!(),
        "log2" => unimplemented!(),
        _ => panic!("Unsupported operation: {}", ops),
    }

    Ok(())
}



