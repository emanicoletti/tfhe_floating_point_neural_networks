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

pub static EXP2_W_CONV_32: [[u32; 4]; 2] = [
    [
        (-0.3530_f32).to_bits(), (-0.1346_f32).to_bits(), (-0.4294_f32).to_bits(), (0.0095_f32).to_bits()
    ],
    [
        (0.3344_f32).to_bits(), (-0.3591_f32).to_bits(), (-0.4967_f32).to_bits(), (-0.0329_f32).to_bits()
    ]
];

pub static EXP2_B_CONV_32: [u32; 2] = [
   (0.4315_f32).to_bits(), (0.3426_f32).to_bits()
];

pub static EXP2_W_FC_32: [[u32; 32]; 3] = [
    [
        (-0.1080_f32).to_bits(), (-0.0467_f32).to_bits(), (-0.1571_f32).to_bits(), (-0.0295_f32).to_bits(),  (0.0575_f32).to_bits(),  (0.0514_f32).to_bits(), (-0.1226_f32).to_bits(), (-0.0744_f32).to_bits(),
        (0.0440_f32).to_bits(),  (0.0313_f32).to_bits(),  (0.1492_f32).to_bits(), (0.0735_f32).to_bits(), (-0.1097_f32).to_bits(), (0.1764_f32).to_bits(), (-0.1467_f32).to_bits(), (-0.0109_f32).to_bits(),
        (-0.1452_f32).to_bits(),  (0.0486_f32).to_bits(), (-0.0444_f32).to_bits(), (-0.0793_f32).to_bits(), (-0.0331_f32).to_bits(),  (0.0478_f32).to_bits(), (-0.1178_f32).to_bits(), (-0.1386_f32).to_bits(),
        (-0.1762_f32).to_bits(), (-0.0290_f32).to_bits(), (-0.0647_f32).to_bits(), (-0.1212_f32).to_bits(),  (0.0568_f32).to_bits(), (-0.1003_f32).to_bits(), (-0.1331_f32).to_bits(),  (0.0803_f32).to_bits()
    ],
    [
        (0.0230_f32).to_bits(), (-0.1334_f32).to_bits(), (0.0799_f32).to_bits(), (0.0548_f32).to_bits(), (0.1597_f32).to_bits(), (-0.0273_f32).to_bits(), (0.0720_f32).to_bits(), (-0.0403_f32).to_bits(),
        (-0.0005_f32).to_bits(),  (0.0950_f32).to_bits(), (-0.0553_f32).to_bits(),  (0.0618_f32).to_bits(), (-0.1023_f32).to_bits(),  (0.1024_f32).to_bits(), (-0.0521_f32).to_bits(), (-0.1109_f32).to_bits(),
        (0.0641_f32).to_bits(), (-0.1506_f32).to_bits(),  (0.0586_f32).to_bits(), (-0.1346_f32).to_bits(), (-0.0253_f32).to_bits(), (-0.0531_f32).to_bits(),  (0.0484_f32).to_bits(), (-0.0186_f32).to_bits(),
        (-0.1399_f32).to_bits(),  (0.1726_f32).to_bits(), (-0.0364_f32).to_bits(),  (0.1599_f32).to_bits(),  (0.0657_f32).to_bits(), (-0.1089_f32).to_bits(),  (0.1140_f32).to_bits(), (-0.0728_f32).to_bits()
    ],
    [
        (0.0179_f32).to_bits(), (-0.1506_f32).to_bits(), (-0.1107_f32).to_bits(), (0.0530_f32).to_bits(), (0.1704_f32).to_bits(), (-0.1007_f32).to_bits(), (-0.1562_f32).to_bits(), (-0.0034_f32).to_bits(),
        (0.0296_f32).to_bits(), (-0.0540_f32).to_bits(), (-0.0647_f32).to_bits(), (0.1399_f32).to_bits(), (-0.0627_f32).to_bits(), (-0.0787_f32).to_bits(), (0.0602_f32).to_bits(), (-0.1117_f32).to_bits(),
        (0.0761_f32).to_bits(), (0.0057_f32).to_bits(), (0.1213_f32).to_bits(), (-0.0973_f32).to_bits(), (-0.1132_f32).to_bits(), (0.0301_f32).to_bits(), (-0.1713_f32).to_bits(), (-0.0555_f32).to_bits(),
        (0.1276_f32).to_bits(), (0.0343_f32).to_bits(), (-0.1384_f32).to_bits(), (-0.1223_f32).to_bits(), (0.1393_f32).to_bits(), (-0.0778_f32).to_bits(), (0.0590_f32).to_bits(), (0.0405_f32).to_bits()
    ]
];

pub static EXP2_B_FC_32: [u32; 3] = [
    (0.0335_f32).to_bits(), (0.0322_f32).to_bits(), (-0.0740_f32).to_bits()
];
