extern crate lazy_static;

mod definitions;
mod erf_cody;
mod lets_be_rational;
mod normal_distribution;
mod rational_cubic;

pub use lets_be_rational::iv_implied_volatility_from_a_transformed_rational_guess;
