/*
 * linearRegression.hpp
 *
 *  Created on: 10/04/2019
 *  Author: C.Marsh
 *
 *  Seeing if I can use Stans MCMC and optimsier's to solve y = a + bX + e
 *  if this is possible I am confident that I can port this into Casal2
 */

#ifndef MODEL_MODEL_LINEAR_REGRESSION_HPP_
#define MODEL_MODEL_LINEAR_REGRESSION_HPP_


#include <stan/model/model_header.hpp>
typedef stan::math::var Double;

namespace niwa {
  namespace model_linear_namespace {
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
      reader.add_event(0, 0, "start", "model_rosenbrock");
      reader.add_event(20, 20, "end", "model_rosenbrock");
      return reader;
    }

/*
    // Mimick the estimate manager.
    class parameters {
    public:
      parameters(vector<Double> start_vals) {};
      virtual                         ~parameters() {};
      Double                      value() { return *target_; }
    protected:
      vector<Double>*                     target_ = nullptr;
    };
*/


    class model_linear { // See if we can override without inheriting from the class prob_grad
    private:
      int N_; // number of observatiosn
      Double sigma_;
      vector<Double> y_; // observerd
      vector<Double> x_; // expected
      // replicate how Casal2 handles parameters so I am confident everything can be done.
      Double         a_;
      Double         b_;
      Double*        a_target_ = &a_;
      Double*        b_target_ = &b_;

    protected:
      size_t num_params_r__;
      std::vector<std::pair<int, int> > param_ranges_i__;
    public:

      // My custom functions
      model_linear(std::ostream* pstream__, unsigned int random_seed__, vector<Double> y, vector<Double> x, Double sigma) :
        y_(y), x_(x), sigma_(sigma),num_params_r__(2),  param_ranges_i__(std::vector<std::pair<int, int> >(0)) {
        my_ctor_body(random_seed__, pstream__),
        N_ = y.size();
      }

      void my_ctor_body(unsigned int random_seed__,  std::ostream* pstream__) {
        boost::ecuyer1988 base_rng__ = stan::services::util::create_rng(random_seed__, 0);
        (void) base_rng__;  // suppress unused var warning
        size_t pos__;
        (void) pos__;  // dummy to suppress unused var warning
        std::vector<int> vals_i__;
        std::vector<double> vals_r__;
      }
      ~model_linear() { }

      inline size_t num_params_r() const {
        return num_params_r__;
      }

      inline size_t num_params_i()  {
        return param_ranges_i__.size();
      }

      inline std::pair<int, int> param_range_i(size_t idx)  {
        return param_ranges_i__[idx];
      }


      /*
       * Thoughts on this method
       * This method transforms parameters from constrained -> unconstrained space taking the values by reference & params_r__
       * And overwriting the unconstrained values as into the reference parameter params_r__, by doing nothing assumes parameters
       * are unconstrained (no transformation required)
       */
      void transform_inits(const stan::io::var_context& context__,
                 std::vector<int>& params_i__,
                 std::vector<double>& params_r__,
                 std::ostream* pstream__)  {

        // input params_r__ and params_i__ and empty context__ with a stream,
        // check with bounds which are local class variables and re-write params_r as
        // un constrainded values.
        std::cerr << "transform_inits - do nothing\n";

      }

      void transform_inits(const stan::io::var_context& context,
                 Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                 std::ostream* pstream__)  {
        std::vector<double> params_r_vec;
        std::vector<int> params_i_vec;
        transform_inits(context, params_i_vec, params_r_vec, pstream__);
        params_r.resize(params_r_vec.size());
        for (int i = 0; i < params_r.size(); ++i)
          params_r(i) = params_r_vec[i];
      }


      vector<Double> calc_sse(vector<Double>& params) const {
        vector<Double> y_hat(N_,0.0);
        Double SSE = 0.0;
        for (unsigned i = 0; i < params.size(); ++i)
          std::cout << params[i] << " ";
        std::cout << "\n";
        for(unsigned i = 0; i < N_; ++i) {
            y_hat[i] = params[0] + params[1] * x_[i];
            //std::cout << "i = " << i << " yhat = " <<  y_hat[i] << " x = " <<  x_[i] <<  " should be " << 0.324 + 1.34 * x_[i] <<  "\n";
            SSE += (y_[i] - y_hat[i]) * y_[i] - y_hat[i];
        }
        return y_hat;
      }

      // More like Casal2's handling of parameters, using pointers
      vector<Double> alt_calc_sse() const {
        vector<Double> y_hat(N_,0.0);
        Double SSE = 0.0;
        for(unsigned i = 0; i < N_; ++i) {
            y_hat[i] = a_ + b_ * x_[i];
            //std::cout << "i = " << i << " yhat = " <<  y_hat[i] << " x = " <<  x_[i] <<  " should be " << 0.324 + 1.34 * x_[i] <<  "\n";
            SSE += (y_[i] - y_hat[i]) * y_[i] - y_hat[i];
        }
        return y_hat;
      }
      // Return Log_prob an easy class to change probably the key class
      template<bool propto__, bool jacobian__, typename T__>
      T__ log_prob(vector<T__>& params_r__, vector<int>& params_i__,
          std::ostream* pstream__ = 0) const {

        stan::math::accumulator<T__> lp_accum__;

        vector<T__> val;
        /*
        vector<double> vals;
        for (unsigned i = 0; i < params_r__.size(); ++i) {
          val = stan::math::value_of(params_r__[i]);
          vals.push_back(val); // the grad function has params_r__ as a vector of stan::math::var objects.
        }
        */
        *a_target_ = params_r__[0];
        *b_target_ = params_r__[1];
        for (unsigned i = 0; i < params_r__.size(); ++i)
          std::cout << params_r__[i] << " ";
        std::cout << "- log_prob\n";

        val = alt_calc_sse();

        //val = calc_sse(params_r__);
        lp_accum__.add(normal_log(y_, val, sigma_));
        return lp_accum__.sum();
        /*
        vector<T__> y_hat(N_,0.0);
        T__ SSE = 0;
        for(unsigned i = 0; i < N_; ++i) {
          std::cout << "i = " << i << " yhat = " <<  y_hat[i] << "\n";
          y_hat[i] = params_r__[0] + params_r__[1] * x_[i];
          SSE += (y_[i] - y_hat[i]) * y_[i] - y_hat[i];
        }
        lp_accum__.add(normal_log(y_, y_hat, sigma_));
        return lp_accum__.sum();
        */
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
      void get_param_names(std::vector<std::string>& names__)  {
        names__.resize(0);
        names__.push_back("a");
        names__.push_back("b");
      }

      void get_dims(std::vector<std::vector<size_t> >& dimss__)  {
        dimss__.resize(0);
        std::vector<size_t> dims__;
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
      }

      template <typename RNG>
      void write_array(RNG& base_rng__,
               std::vector<double>& params_r__,
               std::vector<int>& params_i__,
               std::vector<double>& vars__,
               bool include_tparams__ = true,
               bool include_gqs__ = true,
               std::ostream* pstream__ = 0)  {


      stan::io::reader<double> in__(params_r__, params_i__);
      vector<double> constrained_pars(params_r__.size());

      vars__.resize(0);
      for (int i = 0; i < params_r__.size(); ++i) {
        constrained_pars[i] = in__.scalar_constrain();
      }
      vars__.push_back(constrained_pars[0]);
      vars__.push_back(constrained_pars[1]);
      }

      template <typename RNG>
      void write_array(RNG& base_rng,
               Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
               Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
               bool include_tparams = true,
               bool include_gqs = true,
               std::ostream* pstream = 0)  {
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
        return "model_linear";
      }

      void constrained_param_names(std::vector<std::string>& param_names__,
                     bool include_tparams__ = true,
                     bool include_gqs__ = true)  {
            std::stringstream param_name_stream__;
            param_name_stream__.str(std::string());
            param_name_stream__ << "a";
            param_names__.push_back(param_name_stream__.str());
            param_name_stream__.str(std::string());
            param_name_stream__ << "b";
            param_names__.push_back(param_name_stream__.str());

            return;

      }
      void unconstrained_param_names(std::vector<std::string>& param_names__,
                       bool include_tparams__ = true,
                       bool include_gqs__ = true)  {
        std::stringstream param_name_stream__;
        param_name_stream__.str(std::string());
        param_name_stream__ << "a";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "b";
        param_names__.push_back(param_name_stream__.str());

      }
    }; // model
  } // namespace
}
#endif /* MODEL_MODEL_LINEAR_REGRESSION_HPP_ */
