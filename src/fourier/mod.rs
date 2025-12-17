use std::{
    f32::consts::PI,
    ops::{Div, Mul},
};

use group::ff::{Field, PrimeField};
use num::{One, ToPrimitive};
use num_complex::{Complex32, ComplexFloat, c32};

// input length `l` must be a poswer of 2
pub fn fourier_coeffs_internal_ff<F: Field>(input: &[F], omega: F) -> Vec<F> {
    // not zero and a power of 2
    let n: usize = input.len();

    assert!(n.count_ones() == 1);

    if n == 1 {
        return Vec::from([input[0]]);
    }

    let even: Vec<_> = input.iter().step_by(2).copied().collect();
    let odd: Vec<_> = input.iter().skip(1).step_by(2).copied().collect();

    let omega_squared = omega * omega;
    let even_coeffs = fourier_coeffs_internal_ff(&even, omega_squared);
    let mut odd_coeffs = fourier_coeffs_internal_ff(&odd, omega_squared);

    let mut f = F::ONE;
    odd_coeffs.iter_mut().for_each(|v| {
        *v *= f;
        f *= omega;
    });

    let mut low_part: Vec<_> = even_coeffs
        .iter()
        .zip(odd_coeffs.iter())
        .map(|(x, y)| *x + *y)
        .collect();

    let high_part: Vec<_> = even_coeffs
        .iter()
        .zip(odd_coeffs.iter())
        .map(|(x, y)| *x - *y)
        .collect();

    low_part.extend(high_part);

    low_part
}

fn get_primitive_root<F: PrimeField>(domain_size: usize) -> F {
    assert!(domain_size.count_ones() == 1);

    let cls = domain_size.ilog2();
    let fs = F::S;

    assert!(cls <= fs);

    let mut exp = F::S - cls;

    let mut phi = F::ROOT_OF_UNITY;

    while exp > 0 {
        phi *= phi;

        exp -= 1;
    }

    phi
}

pub fn fourier_coeffs_ff<F: PrimeField>(input: &[F]) -> Vec<F> {
    fourier_coeffs_internal_ff(input, get_primitive_root(input.len()))
}

pub fn fourier_coeffs_inv_ff<F: PrimeField>(input: &[F]) -> Vec<F> {
    let mut coeffs = fourier_coeffs_internal_ff(
        input,
        get_primitive_root::<F>(input.len()).invert().unwrap(),
    );

    let one_by_n = F::from_u128(input.len() as u128).invert().unwrap();

    coeffs.iter_mut().for_each(|v| *v *= one_by_n);

    coeffs
}

// ---

pub fn fourier_coeffs_internal<const IS_INV: bool>(input: &[Complex32]) -> Vec<Complex32> {
    // not zero and a power of 2
    let n: usize = input.len();

    assert!(n.count_ones() == 1);

    if n == 1 {
        return Vec::from([input[0]]);
    }

    let even: Vec<_> = input.iter().step_by(2).copied().collect();
    let odd: Vec<_> = input.iter().skip(1).step_by(2).copied().collect();

    let even_coeffs = fourier_coeffs_internal::<IS_INV>(&even);
    let mut odd_coeffs = fourier_coeffs_internal::<IS_INV>(&odd);

    let phi = if !IS_INV {
        Complex32::cis(-(2.0 * PI) / (n as f32))
    } else {
        Complex32::cis((2.0 * PI) / (n as f32))
    };

    let mut f = Complex32::one();
    odd_coeffs.iter_mut().for_each(|v| {
        *v *= f;
        f *= phi;
    });

    let mut low_part: Vec<_> = even_coeffs
        .iter()
        .zip(odd_coeffs.iter())
        .map(|(x, y)| x + y)
        .collect();

    let high_part: Vec<_> = even_coeffs
        .iter()
        .zip(odd_coeffs.iter())
        .map(|(x, y)| x - y)
        .collect();

    low_part.extend(high_part);

    low_part
}

pub fn fourier_coeffs(input: &[Complex32]) -> Vec<Complex32> {
    fourier_coeffs_internal::<false>(input)
}

pub fn fourier_coeffs_inv(input: &[Complex32]) -> Vec<Complex32> {
    let mut coeffs = fourier_coeffs_internal::<true>(input);

    let one_by_n = 1.0_f32.div(input.len().to_i16().unwrap().to_f32().unwrap());

    coeffs.iter_mut().for_each(|v| *v *= one_by_n);

    coeffs
}

#[derive(Debug)]
pub struct Polynomial {
    coeffs: Vec<f32>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<f32>) -> Self {
        Polynomial { coeffs }
    }
}

#[derive(Debug)]
pub struct PolynomialFFTMult {
    pub p: Polynomial,
}

impl Mul for &PolynomialFFTMult {
    type Output = PolynomialFFTMult;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.p.coeffs.len().count_ones(), 1);
        assert_eq!(self.p.coeffs.len(), rhs.p.coeffs.len());

        let self_len = self.p.coeffs.len();
        let rhs_len = rhs.p.coeffs.len();

        let mut self_extended = self.p.coeffs.clone();
        self_extended.extend(std::iter::repeat_n(0_f32, rhs_len));

        let mut rhs_extended = rhs.p.coeffs.clone();
        rhs_extended.extend(std::iter::repeat_n(0_f32, self_len));

        let as_complex_vec = |vec: &Vec<f32>| {
            vec.iter()
                .map(|v| c32(v.to_f32().unwrap(), 0.0))
                .collect::<Vec<_>>()
        };

        let self_fourier = fourier_coeffs(&as_complex_vec(&self_extended));
        let rhs_fourier = fourier_coeffs(&as_complex_vec(&rhs_extended));

        let pointwise_multiplied: Vec<_> = self_fourier
            .iter()
            .zip(rhs_fourier.iter())
            .map(|(l, r)| l * r)
            .collect();

        let coeffs = fourier_coeffs_inv(&pointwise_multiplied);

        PolynomialFFTMult {
            p: Polynomial {
                coeffs: coeffs.iter().map(|v| v.re()).collect(),
            },
        }
    }
}

impl Mul for &Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = Vec::new();
        result.resize(self.coeffs.len() + rhs.coeffs.len(), 0_f32);

        for (self_idx, self_value) in self.coeffs.iter().enumerate() {
            for (rhs_idx, rhs_value) in rhs.coeffs.iter().enumerate() {
                result[self_idx + rhs_idx] += self_value * rhs_value
            }
        }

        Polynomial { coeffs: result }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_n;

    use ff::PrimeField;
    use num::ToPrimitive;
    use num_complex::c32;

    use crate::fourier::{
        Polynomial, PolynomialFFTMult, fourier_coeffs, fourier_coeffs_ff, fourier_coeffs_inv,
        fourier_coeffs_inv_ff,
    };
    use num_complex::ComplexFloat;

    #[derive(PrimeField)]
    #[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
    #[PrimeFieldGenerator = "7"]
    #[PrimeFieldReprEndianness = "little"]
    struct Fp([u64; 4]);

    #[test]
    fn fourier_complex() {
        let seq: Vec<i16> = repeat_n(0_i16..64, 16).flatten().collect();

        let coeffs = fourier_coeffs(
            &seq.iter()
                .map(|v| c32(v.to_f32().unwrap(), 0.0))
                .collect::<Vec<_>>(),
        );
        let restored_seq = fourier_coeffs_inv(&coeffs)
            .iter()
            .map(|c| c.re().round().to_i16().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(seq, restored_seq)
    }

    #[test]
    fn fourier_ff() {
        let seq: Vec<Fp> = repeat_n(0_i16..64, 16)
            .flatten()
            .map(|n| Fp::from_u128(n as u128))
            .collect();

        let coeffs = fourier_coeffs_ff(&seq);

        let restored_seq = fourier_coeffs_inv_ff(&coeffs);

        assert_eq!(seq, restored_seq)
    }

    #[test]
    fn polynomial_multiplication() {
        let p1 = Polynomial {
            coeffs: vec![1.0, 2.0, 3.0, 4.0],
        };
        let p2 = Polynomial {
            coeffs: vec![5.0, 6.0, 7.0, 8.0],
        };

        let mult1 = &p1 * &p2;

        let mult2 = &PolynomialFFTMult { p: p1 } * &PolynomialFFTMult { p: p2 };

        println!("mult1: {:?}", mult1);

        println!("mult2: {:?}", mult2);
    }
}
