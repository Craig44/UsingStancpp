/*
 * 8schools.hpp
 *
 *  Created on: 18/03/2019
 *      Author: Dell User
 */

#ifndef MODEL_MODEL_8SCHOOLS_HPP_
#define MODEL_MODEL_8SCHOOLS_HPP_


#include <stan/model/model_header.hpp>
namespace niwa {
namespace model_8schools_namespace {
	using std::istream;
	using std::string;
	using std::stringstream;
	using std::vector;
	using stan::io::dump;
	using stan::model::prob_grad;
	using namespace stan::math;

	typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;
	typedef Eigen::Matrix<double,1,Eigen::Dynamic> row_vector_d;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_d;

  #define INF std::numeric_limits<double>::max()

	static int current_statement_begin__;


	stan::io::program_reader prog_reader__() {
		stan::io::program_reader reader;
		reader.add_event(0, 0, "start", "model_8schools");
		reader.add_event(20, 20, "end", "model_8schools");
		return reader;
	}

	class model_8schools : public prob_grad {
	private:
		int J_;
		vector<double> sigma_;
    vector<double> y_;
    vector<double> lower_bound_;
    vector<double> upper_bound_;

	public:
/*		model_8schools(stan::io::var_context& context__,
			std::ostream* pstream__ = 0)
			: prob_grad(0) {
			ctor_body(context__, 0, pstream__);
		}

		model_8schools(stan::io::var_context& context__,
			unsigned int random_seed__,
			std::ostream* pstream__ = 0)
			: prob_grad(0) {
			ctor_body(context__, random_seed__, pstream__);
		}*/

		// My custom functions
    model_8schools(int J, vector<double> sigma, vector<double> y, vector<double> lower_bound,
        vector<double> upper_bound, std::ostream* pstream__, unsigned int random_seed__)
      : prob_grad(lower_bound.size()), J_(J), sigma_(sigma), y_(y), lower_bound_(lower_bound), upper_bound_(upper_bound) {
      my_ctor_body(random_seed__, pstream__);
    }

		void my_ctor_body(unsigned int random_seed__,  std::ostream* pstream__) {
      boost::ecuyer1988 base_rng__ = stan::services::util::create_rng(random_seed__, 0);
      (void) base_rng__;  // suppress unused var warning
      size_t pos__;
      (void) pos__;  // dummy to suppress unused var warning
      std::vector<int> vals_i__;
      std::vector<double> vals_r__;

		}
/*

		// what does this function do?
		// constructor
		// This will be easy to bypass context in this method, custom constructor for the problem now.
		void ctor_body(stan::io::var_context& context__,
					   unsigned int random_seed__,
					   std::ostream* pstream__) {
			boost::ecuyer1988 base_rng__ =
			  stan::services::util::create_rng(random_seed__, 0);
			(void) base_rng__;  // suppress unused var warning

			current_statement_begin__ = -1;

			static const char* function__ = "model_8schools_namespace::model_8schools";
			(void) function__;  // dummy to suppress unused var warning
			size_t pos__;
			(void) pos__;  // dummy to suppress unused var warning
			std::vector<int> vals_i__;
			std::vector<double> vals_r__;
			double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
			(void) DUMMY_VAR__;  // suppress unused var warning

			// initialize member variables
			try {
				current_statement_begin__ = 2;
				context__.validate_dims("data initialization", "J", "int", context__.to_vec());
				J = int(0);
				vals_i__ = context__.vals_i("J");
				pos__ = 0;
				J = vals_i__[pos__++];
				current_statement_begin__ = 3;
				validate_non_negative_index("y", "J", J);
				context__.validate_dims("data initialization", "y", "double", context__.to_vec(J));
				validate_non_negative_index("y", "J", J);
				y = std::vector<double>(J,double(0));
				vals_r__ = context__.vals_r("y");
				pos__ = 0;
				size_t y_limit_0__ = J;
				for (size_t i_0__ = 0; i_0__ < y_limit_0__; ++i_0__) {
					y[i_0__] = vals_r__[pos__++];
				}
				current_statement_begin__ = 4;
				validate_non_negative_index("sigma", "J", J);
				context__.validate_dims("data initialization", "sigma", "double", context__.to_vec(J));
				validate_non_negative_index("sigma", "J", J);
				sigma = std::vector<double>(J,double(0));
				vals_r__ = context__.vals_r("sigma");
				pos__ = 0;
				size_t sigma_limit_0__ = J;
				for (size_t i_0__ = 0; i_0__ < sigma_limit_0__; ++i_0__) {
					sigma[i_0__] = vals_r__[pos__++];
				}

				// validate, data variables
				current_statement_begin__ = 2;
				check_greater_or_equal(function__,"J",J,0);
				current_statement_begin__ = 3;
				current_statement_begin__ = 4;
				for (int k0__ = 0; k0__ < J; ++k0__) {
					check_greater_or_equal(function__,"sigma[k0__]",sigma[k0__],0);
				}
				// initialize data variables


				// validate transformed data

				// validate, set parameter ranges
				num_params_r__ = 0U;
				param_ranges_i__.clear();
				current_statement_begin__ = 7;
				++num_params_r__;
				current_statement_begin__ = 8;
				++num_params_r__;
				current_statement_begin__ = 9;
				validate_non_negative_index("eta", "J", J);
				num_params_r__ += J;
			} catch (const std::exception& e) {
				stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
				// Next line prevents compiler griping about no return
				throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
			}
		}
*/

		~model_8schools() { }

		/*
		 * Thoughts on this method
		 * This method transforms parameters from constrained -> unconstrained space taking the values by reference & params_r__
		 * And saving the unconstrained values as params_r__ using the writer, So we can also copy this functionality pretty easy
		 * For a pre-subscribed problem. generalising might be difficult.
		 */
		void transform_inits(const stan::io::var_context& context__,
							 std::vector<int>& params_i__,
							 std::vector<double>& params_r__,
							 std::ostream* pstream__) const {

		  // input params_r__ and params_i__ and empty context__ with a stream,
		  // check with bounds which are local class variables and re-write params_r as
		  // un constrainded values.

			stan::io::writer<double> writer__(params_r__,params_i__);
			// we can keep this writer__ class it is helpful for constraints.
			for (int i = 0; i < params_r__.size(); ++i) {
			  if ((lower_bound_[i] <= -INF) & (upper_bound_[i] >= INF)) {
			    writer__.scalar_unconstrain(params_r__[i]);
			  } else if (lower_bound_[i] <= -INF & upper_bound_[i] <= INF) {
          writer__.scalar_ub_unconstrain(upper_bound_[i], params_r__[i]);
			  } else if (lower_bound_[i] >= -INF & upper_bound_[i] >= INF) {
          writer__.scalar_lb_unconstrain(lower_bound_[i], params_r__[i]);
        } else {
          writer__.scalar_lub_unconstrain(lower_bound_[i],upper_bound_[i], params_r__[i]);
        }
			}

			params_r__ = writer__.data_r();
			params_i__ = writer__.data_i();
		}

		void transform_inits(const stan::io::var_context& context,
							 Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
							 std::ostream* pstream__) const {
		  std::vector<double> params_r_vec;
		  std::vector<int> params_i_vec;
		  transform_inits(context, params_i_vec, params_r_vec, pstream__);
		  params_r.resize(params_r_vec.size());
		  for (int i = 0; i < params_r.size(); ++i)
		    params_r(i) = params_r_vec[i];
		}


		// Return Log_prob an easy class to change probably the key class
  template<bool propto__, bool jacobian__, typename T__>
  T__ log_prob(vector<T__>& params_r__, vector<int>& params_i__,
      std::ostream* pstream__ = 0) const {
    // account for jacobian, transform parameters, and calculate log likelihood.
    T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning

    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;

    vector<T__> constrained_pars(params_r__.size());
    stan::io::reader<T__> in__(params_r__, params_i__);

    if (not jacobian__) {
      for (int i = 0; i < params_r__.size(); ++i) {
        if ((lower_bound_[i] <= -INF) & (upper_bound_[i] >= INF)) {
          constrained_pars[i] = in__.scalar_constrain();
        } else if (lower_bound_[i] <= -INF & upper_bound_[i] <= INF) {
          constrained_pars[i] = in__.scalar_ub_constrain(upper_bound_[i]);
        } else if (lower_bound_[i] >= -INF & upper_bound_[i] >= INF) {
          constrained_pars[i] = in__.scalar_lb_constrain(lower_bound_[i]);
        } else {
          constrained_pars[i] = in__.scalar_lub_constrain(lower_bound_[i],upper_bound_[i]);
        }
      }
    } else {
      for (int i = 0; i < params_r__.size(); ++i) {
        if ((lower_bound_[i] <= -INF) & (upper_bound_[i] >= INF)) {
          constrained_pars[i] = in__.scalar_constrain(lp__);
        } else if (lower_bound_[i] <= -INF & upper_bound_[i] <= INF) {
          constrained_pars[i] = in__.scalar_ub_constrain(upper_bound_[i], lp__);
        } else if (lower_bound_[i] >= -INF & upper_bound_[i] >= INF) {
          constrained_pars[i] = in__.scalar_lb_constrain(lower_bound_[i], lp__);
        } else {
          constrained_pars[i] = in__.scalar_lub_constrain(lower_bound_[i],upper_bound_[i], lp__);
        }
        //std::cerr << "parameter index = " << i + 1 << " accumultated contribution = " << lp__ << std::endl;
      }
    }

/*
    std::cerr << "lp__ = " << lp__ << std::endl;
    for(auto val : constrained_pars)
      std::cerr << val << " ";
    std::cerr << "\n";
*/

    T__ mu = constrained_pars[0];
    T__ tau = constrained_pars[1];
    vector<T__> eta;
    size_t dim_eta_0__ = J_;
    eta.resize(dim_eta_0__);
    for (size_t k_0__ = 0; k_0__ < dim_eta_0__; ++k_0__) {
      eta[k_0__] = constrained_pars[2 + k_0__];
    }
/*
    std::cerr << "etas " << std::endl;
    for(auto val : eta)
      std::cerr << val << " ";
    std::cerr << "\n";
*/

    vector<T__> theta(J_);
    stan::math::initialize(theta, DUMMY_VAR__);
    stan::math::fill(theta, DUMMY_VAR__);
    for (int j = 0; j < J_; ++j)
      theta[j] = mu + tau * eta[j];

/*    std::cerr << "theta " << std::endl;
    for(auto val : theta)
      std::cerr << val << " ";
    std::cerr << "\n";
    */
    // validate transformed parameters
    for (int i0__ = 0; i0__ < J_; ++i0__) {
      if (stan::math::is_uninitialized(theta[i0__])) {
        std::stringstream msg__;
        msg__ << "Undefined transformed parameter: theta" << '[' << i0__ << ']';
        throw std::runtime_error(msg__.str());
      }
    }

    const char* function__ = "validate transformed params";
    (void) function__;  // dummy to suppress unused var warning
    current_statement_begin__ = 12;
    current_statement_begin__ = 13;

    // model body

    current_statement_begin__ = 18;
    lp_accum__.add(normal_log(eta, 0, 1));
    current_statement_begin__ = 19;
    //std::cerr << "eta contribution " << lp_accum__.sum() << std::endl;

    lp_accum__.add(normal_log(y_, theta, sigma_));
    //std::cerr << "eta + y contribution " << lp_accum__.sum() << std::endl;

    lp_accum__.add(lp__);
    //std::cerr << "total = " << lp_accum__.sum() << std::endl;

    return lp_accum__.sum();

  } // log_prob()

		template <bool propto, bool jacobian, typename T_>
		T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
				   std::ostream* pstream = 0) const {
		  std::vector<T_> vec_params_r;
		  vec_params_r.reserve(params_r.size());
		  for (int i = 0; i < params_r.size(); ++i)
			vec_params_r.push_back(params_r(i));
		  std::vector<int> vec_params_i;
		  return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
		}

		// thes must be for reporting I think can ignore for now, just make sure it
		// modifies the names__ vector to be of same length as params.
		void get_param_names(std::vector<std::string>& names__) const {
			names__.resize(0);
			names__.push_back("mu");
			names__.push_back("tau");
			names__.push_back("eta");
			names__.push_back("A_non_used_container");
			names__.push_back("theta");
		}


		void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
			dimss__.resize(0);
			std::vector<size_t> dims__;
			dims__.resize(0);
			dimss__.push_back(dims__);
			dims__.resize(0);
			dimss__.push_back(dims__);
			dims__.resize(0);
			dims__.push_back(J_);
			dimss__.push_back(dims__);
			dims__.resize(0);
			dimss__.push_back(dims__);
			dims__.resize(0);
			dims__.push_back(J_);
			dimss__.push_back(dims__);
		}

		template <typename RNG>
		void write_array(RNG& base_rng__,
						 std::vector<double>& params_r__,
						 std::vector<int>& params_i__,
						 std::vector<double>& vars__,
						 bool include_tparams__ = true,
						 bool include_gqs__ = true,
						 std::ostream* pstream__ = 0) const {

    vars__.resize(0);
    vector<double> constrained_pars(params_r__.size());
    stan::io::reader<double> in__(params_r__, params_i__);

    for (int i = 0; i < params_r__.size(); ++i) {
      if ((lower_bound_[i] <= -INF) & (upper_bound_[i] >= INF)) {
        constrained_pars[i] = in__.scalar_constrain();
      } else if (lower_bound_[i] <= -INF & upper_bound_[i] <= INF) {
        constrained_pars[i] = in__.scalar_ub_constrain(upper_bound_[i]);
      } else if (lower_bound_[i] >= -INF & upper_bound_[i] >= INF) {
        constrained_pars[i] = in__.scalar_lb_constrain(lower_bound_[i]);
      } else {
        constrained_pars[i] = in__.scalar_lub_constrain(lower_bound_[i],upper_bound_[i]);
      }
    }

    double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning

    double mu = constrained_pars[0];
    double tau = constrained_pars[1];
    vars__.push_back(mu);
    vars__.push_back(tau);

    vector<double> eta;
    size_t dim_eta_0__ = J_;
    eta.resize(J_);
    for (size_t k_0__ = 0; k_0__ < J_; ++k_0__) {
      eta[k_0__] = constrained_pars[2 + k_0__];
      vars__.push_back(constrained_pars[2 + k_0__]);
    }
    vars__.push_back(0.0); // dummy variables
    vector<double> theta(J_);
    stan::math::initialize(theta, DUMMY_VAR__);
    stan::math::fill(theta, DUMMY_VAR__);
    for (int j = 0; j < J_; ++j) {
      theta[j] = mu + tau * eta[j];
      vars__.push_back(theta[j]);
    }
	}


		template <typename RNG>
		void write_array(RNG& base_rng,
						 Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
						 Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
						 bool include_tparams = true,
						 bool include_gqs = true,
						 std::ostream* pstream = 0) const {
		  std::vector<double> params_r_vec(params_r.size());
		  for (int i = 0; i < params_r.size(); ++i)
			params_r_vec[i] = params_r(i);
		  std::vector<double> vars_vec;
		  std::vector<int> params_i_vec;
		  write_array(base_rng,params_r_vec,params_i_vec,vars_vec,include_tparams,include_gqs,pstream);
		  vars.resize(vars_vec.size());
		  for (int i = 0; i < vars.size(); ++i)
		    vars(i) = vars_vec[i];
		}

		static std::string model_name() {
			return "model_8schools";
		}


		void constrained_param_names(std::vector<std::string>& param_names__,
									 bool include_tparams__ = true,
									 bool include_gqs__ = true) const {
			std::stringstream param_name_stream__;
			param_name_stream__.str(std::string());
			param_name_stream__ << "mu";
			param_names__.push_back(param_name_stream__.str());
			param_name_stream__.str(std::string());
			param_name_stream__ << "tau";
			param_names__.push_back(param_name_stream__.str());
			for (int k_0__ = 1; k_0__ <= J_; ++k_0__) {
				param_name_stream__.str(std::string());
				param_name_stream__ << "eta" << '.' << k_0__;
				param_names__.push_back(param_name_stream__.str());
			}

			if (!include_gqs__ && !include_tparams__) return;
			param_name_stream__.str(std::string());
			param_name_stream__ << "A_non_used_container";
			param_names__.push_back(param_name_stream__.str());
			for (int k_0__ = 1; k_0__ <= J_; ++k_0__) {
				param_name_stream__.str(std::string());
				param_name_stream__ << "theta" << '.' << k_0__;
				param_names__.push_back(param_name_stream__.str());
			}

			if (!include_gqs__) return;
		}


		void unconstrained_param_names(std::vector<std::string>& param_names__,
									   bool include_tparams__ = true,
									   bool include_gqs__ = true) const {
			std::stringstream param_name_stream__;
			param_name_stream__.str(std::string());
			param_name_stream__ << "mu";
			param_names__.push_back(param_name_stream__.str());
			param_name_stream__.str(std::string());
			param_name_stream__ << "tau";
			param_names__.push_back(param_name_stream__.str());
			for (int k_0__ = 1; k_0__ <= J_; ++k_0__) {
				param_name_stream__.str(std::string());
				param_name_stream__ << "eta" << '.' << k_0__;
				param_names__.push_back(param_name_stream__.str());
			}

			if (!include_gqs__ && !include_tparams__) return;
			param_name_stream__.str(std::string());
			param_name_stream__ << "A_non_used_container";
			param_names__.push_back(param_name_stream__.str());
			for (int k_0__ = 1; k_0__ <= J_; ++k_0__) {
				param_name_stream__.str(std::string());
				param_name_stream__ << "theta" << '.' << k_0__;
				param_names__.push_back(param_name_stream__.str());
			}

			if (!include_gqs__) return;
		}

	}; // model
} // namespace
}
#endif /* MODEL_MODEL_8SCHOOLS_HPP_ */
