use half::f16;

/* Experiment 1: Layer Initializations */

pub static EXP1_W_FC1_32: [ [u32; 16]; 4 ] = [
            [ 0.1478_f32.to_bits(),  0.2460_f32.to_bits(), (-0.1902_f32).to_bits(),  0.2048_f32.to_bits(), (-0.0094_f32).to_bits(), (-0.1631_f32).to_bits(), (-0.0475_f32).to_bits(), (-0.0258_f32).to_bits(),
                0.0352_f32.to_bits(),  0.0608_f32.to_bits(),  0.2474_f32.to_bits(),  0.2253_f32.to_bits(),  0.1077_f32.to_bits(),  0.0669_f32.to_bits(), (-0.0140_f32).to_bits(),  0.1992_f32.to_bits()],
            [(-0.0941_f32).to_bits(), (-0.1871_f32).to_bits(), (-0.1587_f32).to_bits(),  0.1157_f32.to_bits(), (-0.2498_f32).to_bits(), (-0.1199_f32).to_bits(),  0.1795_f32.to_bits(), (-0.1308_f32).to_bits(),
                0.1570_f32.to_bits(), (-0.1353_f32).to_bits(),  0.1298_f32.to_bits(), (-0.2283_f32).to_bits(),  0.1142_f32.to_bits(),  0.0593_f32.to_bits(),  0.0358_f32.to_bits(), (-0.0192_f32).to_bits()],
            [(-0.2092_f32).to_bits(),  0.2381_f32.to_bits(), (-0.1979_f32).to_bits(), (-0.2127_f32).to_bits(), (-0.1407_f32).to_bits(),  0.1909_f32.to_bits(),  0.0236_f32.to_bits(),  0.0435_f32.to_bits(),
                0.1414_f32.to_bits(),  0.0379_f32.to_bits(), (-0.2027_f32).to_bits(),  0.1000_f32.to_bits(),  0.1482_f32.to_bits(), (-0.1996_f32).to_bits(),  0.1349_f32.to_bits(), (-0.0602_f32).to_bits()],
            [(-0.0248_f32).to_bits(), (-0.0221_f32).to_bits(), (-0.0468_f32).to_bits(),  0.0468_f32.to_bits(),  0.0116_f32.to_bits(),  0.0588_f32.to_bits(), (-0.2422_f32).to_bits(),  0.0707_f32.to_bits(),
                0.1853_f32.to_bits(), (-0.0841_f32).to_bits(),  0.1562_f32.to_bits(), (-0.0972_f32).to_bits(), (-0.1543_f32).to_bits(), (-0.0157_f32).to_bits(),  0.1084_f32.to_bits(), (-0.2480_f32).to_bits()],
            ];

pub static EXP1_W_FC2_32: [[u32; 4]; 2] = [
    [
        (-0.3546_f32).to_bits(),  (0.2355_f32).to_bits(), (-0.2220_f32).to_bits(), (-0.0288_f32).to_bits()
    ],
    [
        (-0.2830_f32).to_bits(), (-0.4757_f32).to_bits(),  (0.1246_f32).to_bits(),  (0.0483_f32).to_bits()
    ]
];

pub static EXP1_W_FC3_32: [[u32; 2]; 3] = [
    [
        (-0.6488_f32).to_bits(), (0.2701_f32).to_bits()
    ],
    [
        (0.1953_f32).to_bits(), (0.1416_f32).to_bits()
    ],
    [
        (-0.1549_f32).to_bits(), (-0.6687_f32).to_bits()
    ]
];

pub static EXP1_B_FC1_32: [u32; 4] = [
    (-0.0879_f32).to_bits(), 0.1680_f32.to_bits(), (-0.1631_f32).to_bits(), (-0.0271_f32).to_bits()
];

pub static EXP1_B_FC2_32: [u32; 2] = [
    0.4002_f32.to_bits(), (-0.0112_f32).to_bits()
];

pub static EXP1_B_FC3_32: [u32; 3] = [
    (-0.1854_f32).to_bits(), (-0.2199_f32).to_bits(), (-0.6619_f32).to_bits()
];


/* Experiment 2: Layer Initializations */

pub static EXP2_W_CONV_32: [[u32; 2]; 2] = [
    [
        0.1053_f32.to_bits(), 0.2747_f32.to_bits()
    ],
    [
        (-0.2619_f32).to_bits(), (-0.0588_f32).to_bits()
    ]
];

pub static EXP2_B_CONV_32: [u32; 1] = [
    0.4239_f32.to_bits()
];

pub static EXP2_W_FC_32: [[u32; 4]; 3] = [
    [
        (-0.3837_f32).to_bits(), (0.3933_f32).to_bits(), (-0.0095_f32).to_bits(), (-0.2881_f32).to_bits()
    ],
    [
        0.0716_f32.to_bits(), (-0.0016_f32).to_bits(), 0.0192_f32.to_bits(), 0.2232_f32.to_bits()
    ],
    [
        0.3044_f32.to_bits(), (-0.2469_f32).to_bits(), (-0.3236_f32).to_bits(), (-0.2331_f32).to_bits()
    ]
];

pub static EXP2_B_FC_32: [u32; 3] = [
    0.4578_f32.to_bits(), (-0.0027_f32).to_bits(), (-0.1728_f32).to_bits()
];
