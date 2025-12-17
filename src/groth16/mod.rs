use ff::PrimeField;
use group::ff::Field;
use group::{Curve, Group};
use pairing::Engine;
use rand_core::RngCore;

use crate::fourier::{fourier_coeffs_ff, fourier_coeffs_inv_ff};

#[derive(Clone, Copy)]
pub enum Column {
    U,
    V,
    W,
}

pub trait R1CS<F: Field> {
    fn get_n(&self) -> usize;
    fn get_m(&self) -> usize;

    fn get(&self, c: Column, i: usize, j: usize) -> F;
}

pub fn verify_r1cs<F: Field, R: R1CS<F>>(input: &Vec<F>, r1cs: &R) -> bool {
    assert_eq!(input.len(), r1cs.get_m() + 1);

    for i in 0..r1cs.get_n() {
        let u = input.iter().enumerate().fold(F::ZERO, |accum, (j, a)| {
            accum + *a * r1cs.get(Column::U, i, j)
        });
        let v = input.iter().enumerate().fold(F::ZERO, |accum, (j, a)| {
            accum + *a * r1cs.get(Column::V, i, j)
        });
        let w = input.iter().enumerate().fold(F::ZERO, |accum, (j, a)| {
            accum + *a * r1cs.get(Column::W, i, j)
        });

        if u * v != w {
            return false;
        }
    }

    true
}

fn get_column<F: Field, C: R1CS<F>>(c: Column, j: usize, r1cs: &C) -> Vec<F> {
    let mut v = Vec::new();
    for i in 0..(r1cs.get_n()) {
        v.push(r1cs.get(c, i, j));
    }

    v
}

pub struct LagrangeBaseEvaluation<'a, F: Field> {
    eval_point: F,
    l_at_point: F,
    domain: &'a Domain<F>,
}

pub struct Domain<F: Field> {
    pub points: Vec<F>,
}

// `domain_size` must be a power of two
// the function creates a domain from powers of the `domain_size`th primive root
fn create_fft_compatible_domain<F: PrimeField>(domain_size: usize) -> Domain<F> {
    let phi = get_primitive_root::<F>(domain_size);

    let mut points = Vec::new();

    let mut point = F::ONE;
    for _ in 0..domain_size {
        points.push(point);
        point *= phi;
    }

    Domain { points }
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

impl<'a, F: Field> LagrangeBaseEvaluation<'a, F> {
    fn create(domain: &'a Domain<F>, eval_point: F) -> Self {
        let mut l_at_point = F::ONE;
        for domain_point in &domain.points {
            l_at_point *= eval_point - domain_point;
        }

        LagrangeBaseEvaluation {
            eval_point,
            l_at_point,
            domain,
        }
    }

    // argument starts from 0
    // TODO: implement caching of returned values
    fn get_value(&self, i_th_base_polynomial: usize) -> F {
        assert!(i_th_base_polynomial < self.domain.points.len());

        let domain_point_i_th = self.domain.points[i_th_base_polynomial];
        let mut w_i = F::ONE;
        for (index, domain_point) in self.domain.points.iter().enumerate() {
            if index != i_th_base_polynomial {
                w_i *= domain_point_i_th - domain_point;
            }
        }
        w_i = w_i.invert().unwrap();

        let value =
            (self.l_at_point * w_i) * (self.eval_point - domain_point_i_th).invert().unwrap();

        value
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Polynomial<F: Field> {
    // a_0, ..., a_{n-1}
    coeffs: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    pub fn create(coeffs: Vec<F>) -> Self {
        Polynomial { coeffs }
    }

    pub fn evaluate(&self, point: F) -> F {
        let mut r = F::ZERO;
        let mut x_n = F::ONE;

        for coeff in &self.coeffs {
            r += *coeff * x_n;
            x_n *= point;
        }

        r
    }

    pub fn differentiate(&self) -> Self
    where
        F: PrimeField,
    {
        let coeffs = self
            .coeffs
            .iter()
            .skip(1)
            .enumerate()
            .map(|(index, coeff)| F::from_u128((index + 1) as u128) * *coeff)
            .collect();

        Polynomial { coeffs }
    }
}

#[derive(Debug)]
pub struct ProvingKey<G1, G2> {
    alpha_1: G1,
    beta_1: G1,
    beta_2: G2,
    delta_1: G1,
    delta_2: G2,
    u_tau_1: Vec<G1>,
    v_tau_1: Vec<G1>,
    v_tau_2: Vec<G2>,
    comb_witness_1: Vec<G1>,
    lagrange_mod_tau_1: Vec<G1>,
}
#[derive(Debug)]
pub struct VerifyingKey<G1, G2> {
    alpha_1: G1,
    beta_2: G2,
    gamma_2: G2,
    delta_2: G2,
    comb_public_1: Vec<G1>,
}

#[derive(Debug)]
pub struct Proof<G1, G2> {
    a_1: G1,
    b_2: G2,
    c_1: G1,
}

fn evaluate_colum<F: PrimeField, R: R1CS<F>>(c: Column, j: usize, r1cs: &R, point: F) -> F {
    Polynomial::create(fourier_coeffs_inv_ff(&get_column(c, j, r1cs))).evaluate(point)
}

pub fn setup<E: Engine, R: R1CS<E::Fr>>(
    r1cs: &R,
    num_of_public_inputs: usize,
    rng: impl RngCore + Clone,
) -> (ProvingKey<E::G1, E::G2>, VerifyingKey<E::G1, E::G2>)
where
{
    // trapdoor
    let tau = E::Fr::random(rng.clone());
    let alpha = E::Fr::random(rng.clone());
    let beta = E::Fr::random(rng.clone());
    let delta = E::Fr::random(rng.clone());
    let gamma = E::Fr::random(rng.clone());

    let delta_inv = delta.invert().unwrap();
    let gamma_inv = gamma.invert().unwrap();

    let m_whole_range = 0..(r1cs.get_m() + 1);
    let m_public_range = 0..(num_of_public_inputs + 1);
    let m_witness_range = (num_of_public_inputs + 1)..(r1cs.get_m() + 1);

    let u_j_at_tau: Vec<_> = m_whole_range
        .clone()
        .map(|j| evaluate_colum(Column::U, j, r1cs, tau))
        .collect();

    let v_j_at_tau: Vec<_> = m_whole_range
        .clone()
        .map(|j| evaluate_colum(Column::V, j, r1cs, tau))
        .collect();

    let w_j_at_tau: Vec<_> = m_whole_range
        .clone()
        .map(|j| evaluate_colum(Column::W, j, r1cs, tau))
        .collect();

    let g1 = E::G1::generator();
    let g2 = E::G2::generator();

    let alpha_1 = g1 * alpha;
    let beta_1 = g1 * beta;
    let beta_2 = g2 * beta;
    let delta_1 = g1 * delta;
    let delta_2 = g2 * delta;
    let gamma_2 = g2 * gamma;

    let tau_pow_n = tau.pow_vartime(&[r1cs.get_n() as u64]);

    let domain = create_fft_compatible_domain::<E::Fr>(r1cs.get_n());
    let lagrange = LagrangeBaseEvaluation::create(&domain, tau);

    let u_tau_v_tau_1: Vec<_> = m_whole_range
        .clone()
        .map(|i| (g1 * u_j_at_tau[i], g1 * v_j_at_tau[i]))
        .collect();

    let proving_key = ProvingKey {
        alpha_1,
        beta_1,
        beta_2,
        delta_1,
        delta_2,
        u_tau_1: u_tau_v_tau_1.iter().map(|(u, _)| *u).collect(),
        v_tau_1: u_tau_v_tau_1.iter().map(|(_, v)| *v).collect(),
        v_tau_2: m_whole_range.clone().map(|i| g2 * v_j_at_tau[i]).collect(),
        comb_witness_1: m_witness_range
            .clone()
            .map(|i| {
                g1 * ((beta * u_j_at_tau[i] + alpha * v_j_at_tau[i] + w_j_at_tau[i]) * delta_inv)
            })
            .collect(),
        lagrange_mod_tau_1: (0..(r1cs.get_n() - 1))
            .map(|i| g1 * ((lagrange.get_value(i) * (tau_pow_n - E::Fr::ONE)) * delta_inv))
            .collect(),
    };

    let verifying_key = VerifyingKey {
        alpha_1,
        beta_2,
        gamma_2,
        delta_2,
        comb_public_1: m_public_range
            .clone()
            .map(|i| {
                g1 * ((beta * u_j_at_tau[i] + alpha * v_j_at_tau[i] + w_j_at_tau[i]) * gamma_inv)
            })
            .collect(),
    };

    (proving_key, verifying_key)
}

fn compute_h<E: Engine, R: R1CS<E::Fr>>(input: &Vec<E::Fr>, r1cs: &R) -> Vec<E::Fr> {
    let n_whole_range = 0..(r1cs.get_n());
    let n = r1cs.get_n();
    let n_as_field_element = E::Fr::from_u128(n as u128);

    let poly_for = |c: Column| -> Vec<_> {
        n_whole_range
            .clone()
            .map(|n| {
                input
                    .iter()
                    .enumerate()
                    .fold(E::Fr::ZERO, |accum, (index, a)| {
                        accum + *a * r1cs.get(c, n, index)
                    })
            })
            .collect()
    };

    let u = poly_for(Column::U);
    let v = poly_for(Column::V);
    let w = poly_for(Column::W);

    let comput_derivative_evaluations = |vec: &Vec<E::Fr>| {
        let mut t = Polynomial::create(fourier_coeffs_inv_ff(&vec)).differentiate();
        t.coeffs.push(E::Fr::ZERO);
        fourier_coeffs_ff(&t.coeffs)
    };

    let u_diff = comput_derivative_evaluations(&u);
    let v_diff = comput_derivative_evaluations(&v);
    let w_diff = comput_derivative_evaluations(&w);

    let mut h = Vec::new();
    let omega_inv = get_primitive_root::<E::Fr>(n).invert().unwrap();
    let mut omega_inv_power = E::Fr::ONE;
    for index in 0..(r1cs.get_n() - 1) {
        let mut eval = u_diff[index] * v[index] + u[index] * v_diff[index] - w_diff[index];
        let h_denominator = (n_as_field_element * omega_inv_power).invert().unwrap();
        eval *= h_denominator;
        h.push(eval);

        omega_inv_power *= omega_inv;
    }

    h
}

pub fn prove<E: Engine, R: R1CS<E::Fr>>(
    r1cs: &R,
    pk: ProvingKey<E::G1, E::G2>,
    input: Vec<E::Fr>,
    num_of_public_inputs: usize,
    rng: impl RngCore + Clone,
) -> Proof<E::G1, E::G2>
where
{
    assert_eq!(r1cs.get_m() + 1, input.len());

    let r = E::Fr::random(rng.clone());
    let s = E::Fr::random(rng.clone());

    let a_1 = pk.alpha_1
        + pk.delta_1 * r
        + pk.u_tau_1
            .iter()
            .zip(input.iter())
            .fold(E::G1::identity(), |acc, (u, a)| acc + *u * *a);

    let b_1 = pk.beta_1
        + pk.delta_1 * s
        + pk.v_tau_1
            .iter()
            .zip(input.iter())
            .fold(E::G1::identity(), |acc, (v, a)| acc + *v * *a);

    let b_2 = pk.beta_2
        + pk.delta_2 * s
        + pk.v_tau_2
            .iter()
            .zip(input.iter())
            .fold(E::G2::identity(), |acc, (v, a)| acc + *v * *a);

    let h: Vec<E::Fr> = compute_h::<E, _>(&input, r1cs);

    let c_1 = pk
        .comb_witness_1
        .iter()
        .zip(input.iter().skip(num_of_public_inputs + 1))
        .fold(E::G1::identity(), |acc, (c, a)| acc + *c * *a)
        + pk.lagrange_mod_tau_1
            .iter()
            .zip(h.iter())
            .take(r1cs.get_n() - 1)
            .fold(E::G1::identity(), |acc, (c, a)| acc + *c * *a)
        + a_1 * s
        + b_1 * r
        - pk.delta_1 * (r * s);

    Proof { a_1, b_2, c_1 }
}

pub fn verify<E: Engine>(
    vk: VerifyingKey<E::G1, E::G2>,
    public_input: Vec<E::Fr>,
    proof: Proof<E::G1, E::G2>,
) -> bool {
    assert_eq!(public_input.len(), vk.comb_public_1.len());

    let alpha_beta_t = E::pairing(&vk.alpha_1.to_affine(), &vk.beta_2.to_affine());

    let inner_product = vk
        .comb_public_1
        .iter()
        .zip(public_input.iter())
        .fold(E::G1::identity(), |accum, (c, a)| accum + *c * *a);

    let lhs = alpha_beta_t
        + E::pairing(&inner_product.to_affine(), &vk.gamma_2.to_affine())
        + E::pairing(&proof.c_1.to_affine(), &vk.delta_2.to_affine());

    let rhs = E::pairing(&proof.a_1.to_affine(), &proof.b_2.to_affine());

    println!("lhs: {:?}", lhs);
    println!("rhs: {:?}", rhs);

    lhs == rhs
}

#[cfg(test)]
mod tests {
    use bls12_381::Bls12;
    use ff::{Field, PrimeField};
    use group::Curve;
    use group::Group;
    use pairing::Engine;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::groth16::Domain;
    use crate::groth16::LagrangeBaseEvaluation;
    use crate::groth16::create_fft_compatible_domain;
    use crate::groth16::{
        Column, Polynomial, R1CS, get_primitive_root, prove, setup, verify, verify_r1cs,
    };

    struct TestR1CS {
        u: [[<Bls12 as Engine>::Fr; 8]; 4],
        v: [[<Bls12 as Engine>::Fr; 8]; 4],
        w: [[<Bls12 as Engine>::Fr; 8]; 4],
    }

    impl TestR1CS {
        // R1CS (Rank-1 Constraint System) Example
        // ========================================

        // Problem: Verify that given inputs satisfy a polynomial equation system.

        // Input vector (length 5): [1, x1, x2, x3, x4]
        // - First element is always 1 (constant term)
        // - x1 to x4 are the actual inputs

        // We'll add intermediate variables (witnesses):
        // - w1, w2, w3 for intermediate computations

        // Full witness vector: [1, x1, x2, x3, x4, w1, w2, w3]
        //                      [0   1   2   3   4   5   6   7]

        // Computation being verified (3 constraints):
        // 1. w1 = x1 * x2          (multiply first two inputs)
        // 2. w2 = x3 * x4          (multiply next two inputs)
        // 3. w3 = w1 + w2          (sum the products: w3 = w1 + w2)

        fn create() -> Self {
            let mut a = [[<<Bls12 as Engine>::Fr as Field>::ZERO; 8]; 4];
            let mut b = [[<<Bls12 as Engine>::Fr as Field>::ZERO; 8]; 4];
            let mut c = [[<<Bls12 as Engine>::Fr as Field>::ZERO; 8]; 4];

            let one = <<Bls12 as Engine>::Fr as Field>::ONE;

            // # Constraint 1: x1 * x2 = w1
            a[0][1] = one; // x1
            b[0][2] = one; // x2
            c[0][5] = one; // w1

            // # Constraint 2: x3 * x4 = w2
            a[1][3] = one; // x3
            b[1][4] = one; // x4
            c[1][6] = one; // w2

            // # Constraint 3: (w1 + w2) * 1 = w3
            // # This implements addition using multiplication by 1
            a[2][5] = one; // w1
            a[2][6] = one; // w2
            b[2][0] = one; // constant 1
            c[2][7] = one; // w3

            a[3][0] = one;
            b[3][0] = one;
            c[3][0] = one;

            TestR1CS { u: a, v: b, w: c }
        }
    }

    impl R1CS<<Bls12 as Engine>::Fr> for TestR1CS {
        fn get_n(&self) -> usize {
            4
        }

        fn get_m(&self) -> usize {
            7
        }

        fn get(&self, c: Column, i: usize, j: usize) -> <Bls12 as Engine>::Fr {
            match c {
                Column::U => self.u[i][j],
                Column::V => self.v[i][j],
                Column::W => self.w[i][j],
            }
        }
    }

    #[test]
    fn get_primitive_root_test() {
        type F = <Bls12 as Engine>::Fr;

        let omega_4 = get_primitive_root::<F>(4);
        for i in 1..4 {
            assert_ne!(omega_4.pow_vartime(&[i, 0, 0, 0]), F::ONE);
        }
        assert_eq!(omega_4.pow_vartime(&[4, 0, 0, 0]), F::ONE);

        let omega_8 = get_primitive_root::<F>(8);
        for i in 1..8 {
            assert_ne!(omega_8.pow_vartime(&[i, 0, 0, 0]), F::ONE);
        }
        assert_eq!(omega_8.pow_vartime(&[8, 0, 0, 0]), F::ONE);

        let omega_128 = get_primitive_root::<F>(128);
        for i in 1..128 {
            assert_ne!(omega_128.pow_vartime(&[i, 0, 0, 0]), F::ONE);
        }
        assert_eq!(omega_128.pow_vartime(&[128, 0, 0, 0]), F::ONE);
    }

    #[test]
    fn end_2_end_test() {
        let rng = StdRng::from_seed([0u8; 32]);
        let r1cs = TestR1CS::create();

        let input: Vec<<Bls12 as Engine>::Fr> = [1_u32, 1, 2, 3, 4, 2, 12, 14]
            .iter()
            .map(|c| <Bls12 as Engine>::Fr::from_u128(*c as u128))
            .collect();

        assert!(verify_r1cs(&input, &r1cs));

        let number_of_public_inputs = 4;

        let (pk, vk) = setup::<Bls12, _>(&r1cs, number_of_public_inputs, rng.clone());

        let proof = prove::<Bls12, _>(
            &r1cs,
            pk,
            input.clone(),
            number_of_public_inputs,
            rng.clone(),
        );

        let public_input = [1_u32, 1, 2, 3, 4]
            .iter()
            .map(|c| <Bls12 as Engine>::Fr::from_u128(*c as u128))
            .collect();

        assert!(verify::<Bls12>(vk, public_input, proof))
    }

    #[test]
    fn pairing_test() {
        type G1 = <Bls12 as Engine>::G1;
        type G2 = <Bls12 as Engine>::G2;
        type Gt = <Bls12 as Engine>::Gt;
        type Fr = <Bls12 as Engine>::Fr;

        assert_eq!(
            Bls12::pairing(
                &<G1 as Group>::generator().to_affine(),
                &<G2 as Group>::generator().to_affine()
            ),
            <Gt as Group>::generator()
        );

        assert_eq!(
            Bls12::pairing(
                &(<G1 as Group>::generator() * Fr::from_u128(3)).to_affine(),
                &(<G2 as Group>::generator() * Fr::from_u128(2)).to_affine()
            ),
            Bls12::pairing(
                &(<G1 as Group>::generator() * Fr::from_u128(6)).to_affine(),
                &(<G2 as Group>::generator() * Fr::from_u128(1)).to_affine()
            )
        );
    }

    #[test]
    fn differentiate_test() {
        // 1 + 3x + 5x^2 + 7x^3
        // 3 + 10x + 21x^2
        assert_eq!(
            Polynomial {
                coeffs: vec![1, 3, 5, 7]
                    .iter()
                    .map(|n| <Bls12 as Engine>::Fr::from_u128(*n))
                    .collect()
            }
            .differentiate(),
            Polynomial {
                coeffs: vec![3, 10, 21]
                    .iter()
                    .map(|n| <Bls12 as Engine>::Fr::from_u128(*n))
                    .collect()
            }
        );
    }

    #[test]
    fn lagrange_test() {
        type Fr = <Bls12 as Engine>::Fr;

        let polynomial = Polynomial {
            coeffs: vec![1, 3, 5, 7].iter().map(|n| Fr::from_u128(*n)).collect(),
        };

        let eval_point = Fr::from_u128(123456);

        let eval_directly = polynomial.evaluate(eval_point);

        let domain_size = 16;
        let domain = create_fft_compatible_domain::<Fr>(domain_size);
        let lagrange = LagrangeBaseEvaluation::create(&domain, eval_point);

        let base_coeffs: Vec<_> = domain
            .points
            .iter()
            .map(|w_i| polynomial.evaluate(*w_i))
            .collect();

        let eval_with_lagrange_base = base_coeffs
            .iter()
            .enumerate()
            .fold(Fr::ZERO, |accum, (index, coeff)| {
                accum + *coeff * lagrange.get_value(index)
            });

        assert_eq!(eval_directly, eval_with_lagrange_base);
    }
}
