use crate::utils::*;
use std::sync::atomic::{AtomicUsize, Ordering};

//static TWO_PI: f64 = 6.283185307179586476925286766559005768394338798750; //

static VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC: f64 = f64::MIN;
static VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM: f64 = f64::MAX;

/// (DBL_DIG*20)/3 ≈ 100.
/// Only needed when the iteration effectively alternates Householder/Halley/Newton steps
/// and binary nesting due to roundoff truncation.
static IMPLIED_VOLATILITY_MAXIMUM_ITERATIONS: AtomicUsize = AtomicUsize::new(2);

fn get_implied_volatility_maximum_iterations() -> usize {
  IMPLIED_VOLATILITY_MAXIMUM_ITERATIONS.load(Ordering::Relaxed)
}

#[cfg(feature = "ENABLE_SWITCHING_THE_OUTPUT_TO_ITERATION_COUNT")]
static IMPLIED_VOLATILITY_OUTPUT_TYPE: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "ENABLE_SWITCHING_THE_OUTPUT_TO_ITERATION_COUNT")]
fn get_implied_volatility_output_type() -> usize {
  IMPLIED_VOLATILITY_OUTPUT_TYPE.load(Ordering::Relaxed)
}

#[cfg(feature = "ENABLE_SWITCHING_THE_OUTPUT_TO_ITERATION_COUNT")]
fn implied_volatility_output(count: usize, volatility: f64) -> f64 {
  sel(get_implied_volatility_output_type() > 0, count as f64, volatility)
}
#[cfg(not(feature = "ENABLE_SWITCHING_THE_OUTPUT_TO_ITERATION_COUNT"))]
fn implied_volatility_output(count: usize, volatility: f64) -> f64 {
  volatility
}

/// See http://en.wikipedia.org/wiki/Householder%27s_method for a detailed explanation of the third order Householder iteration.
///
/// Given the objective function g(s) whose root x such that 0 = g(s) we seek, iterate
///
///     s_n+1  =  s_n  -  (g/g') · [ 1 - (g''/g')·(g/g') ] / [ 1 - (g/g')·( (g''/g') - (g'''/g')·(g/g')/6 ) ]
///
/// Denoting  newton:=-(g/g'), halley:=(g''/g'), and hh3:=(g'''/g'), this reads
///
///     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]
///
///
/// NOTE that this function returns 0 when beta<intrinsic without any safety checks.
///
fn unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta: f64, x: f64, q: f64 /* q=±1 */, n: usize) -> f64 {
  println!("{:30.13}{:30.13}{:10.1}{:>10}", beta, x, q, n);
  0.0
}

pub fn iv_implied_volatility_from_a_transformed_rational_guess(price: f64, f: f64, k: f64, t: f64, q: f64 /* q=±1 */) -> f64 {
  implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price, f, k, t, q, get_implied_volatility_maximum_iterations())
}

fn implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(mut price: f64, f: f64, k: f64, t: f64, mut q: f64 /* q=±1 */, n: usize) -> f64 {
  let intrinsic = fabs(max(sel(q < 0.0, k - f, f - k), 0.0));
  if price < intrinsic {
    return implied_volatility_output(0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC);
  }
  let max_price = sel(q < 0.0, k, f);
  if price >= max_price {
    return implied_volatility_output(0, VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM);
  }
  let x = (f / k).ln();
  // Map in-the-money to out-of-the-money
  if q * x > 0.0 {
    price = fabs(max(price - intrinsic, 0.0));
    q = -q;
  }
  unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price / (f.sqrt() * k.sqrt()), x, q, n) / t.sqrt()
}
