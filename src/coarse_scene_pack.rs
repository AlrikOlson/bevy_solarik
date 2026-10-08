use crate::coarse_cache::Directional;

/// 76 bytes, including all source observations. No float quantization.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PackedDirectional {
    pub words: [u32; 19],
}
impl PackedDirectional {
    pub fn new(bin: &Directional, side: u32) -> Option<Self> {
        if !bin.validate(side)
            || bin.diagonal[3].to_bits() != 0
            || bin.cross[3].to_bits() != 0
            || bin.normal[3].to_bits() != 0
        {
            return None;
        }
        let mut words = [0; 19];
        words[0] = bin.counts[0] | (bin.counts[1] << 16);
        words[1] = bin.counts[2];
        words[2..6].copy_from_slice(&bin.depth.map(f32::to_bits));
        for i in 0..3 {
            words[6 + i] = bin.diagonal[i].to_bits();
            words[9 + i] = bin.cross[i].to_bits();
            words[12 + i] = bin.normal[i].to_bits();
        }
        for i in 0..4 {
            words[15 + i] = bin.materials[2 * i] | (bin.materials[2 * i + 1] << 16);
        }
        Some(Self { words })
    }
    pub fn unpack(&self) -> Directional {
        let w = &self.words;
        let mut bin = Directional {
            counts: [w[0] & 65535, w[0] >> 16, w[1], 0],
            depth: [w[2], w[3], w[4], w[5]].map(f32::from_bits),
            diagonal: [
                f32::from_bits(w[6]),
                f32::from_bits(w[7]),
                f32::from_bits(w[8]),
                0.,
            ],
            cross: [
                f32::from_bits(w[9]),
                f32::from_bits(w[10]),
                f32::from_bits(w[11]),
                0.,
            ],
            normal: [
                f32::from_bits(w[12]),
                f32::from_bits(w[13]),
                f32::from_bits(w[14]),
                0.,
            ],
            materials: [0; 8],
        };
        for i in 0..8 {
            bin.materials[i] = (w[15 + i / 2] >> ((i % 2) * 16)) & 65535;
        }
        bin
    }
}
