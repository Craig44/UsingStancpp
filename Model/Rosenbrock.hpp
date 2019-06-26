/*
 * Rosenbrock.hpp
 *
 *  Created on: 10/04/2019
 *   Author: C.Marsh
 */


#ifndef MODEL_MODEL_ROSENBROCK_HPP_
#define MODEL_MODEL_ROSENBROCK_HPP_


#include <stan/model/model_header.hpp>

namespace model_rosenbrock_namespace {
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

  class model_rosenbrock { // See if we can override without inheriting from the class prob_grad
  private:

  protected:
    size_t num_params_r__;
    std::vector<std::pair<int, int> > param_ranges_i__;
  public:

    // My custom functions
    model_rosenbrock(std::ostream* pstream__, unsigned int random_seed__) : num_params_r__(2),
    param_ranges_i__(std::vector<std::pair<int, int> >(0)) {
      my_ctor_body(random_seed__, pstream__), ;
    }

    void my_ctor_body(unsigned int random_seed__,  std::ostream* pstream__) {
      boost::ecuyer1988 base_rng__ = stan::services::util::create_rng(random_seed__, 0);
      (void) base_rng__;  // suppress unused var warning
      size_t pos__;
      (void) pos__;  // dummy to suppress unused var warning
      std::vector<int> vals_i__;
      std::vector<double> vals_r__;
    }
    ~model_rosenbrock() { }

    inline size_t num_params_r() const {
      return num_params_r__;
    }

    inline size_t num_params_i() const {
      return param_ranges_i__.size();
    }

    inline std::pair<int, int> param_range_i(size_t idx) const {
      return param_ranges_i__[idx];
    }


    void transform_inits(const stan::io::var_context& context__,
               std::vector<int>& params_i__,
               std::vector<double>& params_r__,
               std::ostream* pstream__) const {

      // input params_r__ and params_i__ and empty context__ with a stream,
      // check with bounds which are local class variables and re-write params_r as
      // un constrainded values.
      std::cerr << "transform_inits - do nothing\n";

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


      T__ lp__(0.0);
      lp__ = 100 * pow((params_r__[1] - params_r__[0] * params_r__[0]),2) + (1 - params_r__[0]) * (1 - params_r__[0]);
      stan::math::accumulator<T__> lp_accum__;
      lp_accum__.add(lp__);
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
      names__.push_back("x1");
      names__.push_back("x2");
    }

    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
      dimss__.resize(0);
      std::vector<size_t> dims__;
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
      return "model_rosenbock";
    }

    void constrained_param_names(std::vector<std::string>& param_names__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true) const {

    }
    void unconstrained_param_names(std::vector<std::string>& param_names__,
                     bool include_tparams__ = true,
                     bool include_gqs__ = true) const {

    }
  }; // model
} // namespace

#endif /* MODEL_MODEL_ROSENBROCK_HPP_ */
