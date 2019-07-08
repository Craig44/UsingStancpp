//============================================================================
// Name        : SimpleModel.cpp
// Author      : C.Marsh
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <sstream>

#include <stan/model/model_header.hpp>
#include <stan/model/finite_diff_grad.hpp>

#include <stan/io/empty_var_context.hpp>

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>

#include <stan/optimization/bfgs.hpp>

#include <stan/services/util/initialize.hpp>
#include <stan/services/util/run_sampler.hpp>

#include <test/unit/services/instrumented_callbacks.hpp>

#include <stan/mcmc/sample.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>

#include <boost/random/uniform_real_distribution.hpp>

#include "model_8schools.hpp"
#include "linearRegression.hpp"
#include "TemplateTest.h"
#include "AltTemplateTest.hpp"
#include <map>
#define INF std::numeric_limits<double>::max()

using namespace niwa;
using namespace model_8schools_namespace;
using namespace model_linear_namespace;

using std::cout;
using std::endl;
using std::map;

typedef stan::math::var Double;


int main() {
	std::cout << "!!!Hello World!!!" << std::endl;
	// Set up class
	stan::io::empty_var_context context;
  unsigned int random_seed = 123;
  int J = 8;
  vector<string> names = {"mu", "tau", "theta1","theta2","theta3","theta4","theta5","theta6","theta7","theta8"};

  vector<double> y = {28,  8, -3,  7, -1,  1, 18, 12};
  vector<double> sigma = {15, 10, 16, 11,  9, 11, 10, 18};
  vector<double> lower_bound{-INF, 0, -INF,-INF,-INF,-INF,-INF,-INF,-INF,-INF};
  vector<double> upper_bound{INF, INF, INF, INF, INF, INF, INF, INF, INF, INF};
  model_8schools my_8schools_model(J, sigma, y, lower_bound, upper_bound, &std::cout, random_seed);
  std::cout << "model name = " << my_8schools_model.model_name() << std::endl;

  // use MLE estimates from Stan
  vector<double> params_r{9.694172, 9362.184, 2.236103e-03,  1.000133e-04, -1.074730e-03, -6.707788e-06, -8.611242e-04, -6.474332e-04,  1.168012e-03,  5.272521e-04};
  //vector<double> start_vals{5, 6000, 0, 0, 0, 0, 0, 0, 0, 0};
  vector<double> start_vals{1.696269, 0.1683022, 0.0205084845,  0.0109040206, -0.0026786834,  0.0078936656, -0.0053568727, -0.0000117153, 0.0289505637,  0.0055900884};

  /*
  double init_radius = 2.0;
  stan::callbacks::logger logger;
  stan::callbacks::writer init_writer;

  std::vector<double> cont_vector
      = stan::services::util::initialize<false>(my_8schools_model, context, random_seed, init_radius, false,
                                logger, init_writer);
  */





  vector<double> unconstrained_params_r = params_r;
  vector<double> unconstrained_start_vals = start_vals;
  vector<int> params_i(0);
  vector<double> gradient;
  std::ostream* msgs = 0;

  /////////////////////////////////////////////////
  // Re-implement a rough version of initialize
  ////////////////////////////////////////////////
  double init_radius = 2.0;
  std::vector<double> cont_vector(params_r.size());
  boost::random::uniform_real_distribution<double> unif(-init_radius, init_radius);
  boost::ecuyer1988 base_rng__ = stan::services::util::create_rng(random_seed, 0);
  int N_inits = 100;
  for (int j = 0; j < N_inits; ++j) {
    for (size_t n = 0; n < params_r.size(); ++n)
      cont_vector[n] = unif(base_rng__);
    // we evaluate this with propto=true since we're
    // evaluating with autodiff variables
    double log_p_grad = stan::model::log_prob_grad<false, true, model_8schools>(my_8schools_model, cont_vector, params_i, gradient, msgs);
    bool gradient_ok = boost::math::isfinite(stan::math::sum(gradient));
    // found a candidate lets run with it
    if (gradient_ok)
      break;
  }

  ////////////////////////
  // unconstrain params and
  // calculate log posterior
  ////////////////////////
  my_8schools_model.transform_inits(context, params_i, unconstrained_params_r, msgs);
  my_8schools_model.transform_inits(context, params_i, unconstrained_start_vals, msgs);

  cout << "transformed variables are" << endl;
  for (auto & name : names)
    cout << name << " ";
  cout << "\n";
  for (auto & val : unconstrained_params_r)
    cout << val << " ";
  cout << "\n\n";

  ////////////// Re-transform parameters
  vector<double> constrained_pars(unconstrained_params_r.size());
  stan::io::reader<double> in__(unconstrained_params_r, params_i);

  cout << "re-transformed variables are" << endl;
  for (int i = 0; i < unconstrained_params_r.size(); ++i) {
    if ((lower_bound[i] <= -INF) & (upper_bound[i] >= INF)) {
      constrained_pars[i] = in__.scalar_constrain();
    } else if (lower_bound[i] <= -INF & upper_bound[i] <= INF) {
      constrained_pars[i] = in__.scalar_ub_constrain(upper_bound[i]);
    } else if (lower_bound[i] >= -INF & upper_bound[i] >= INF) {
      constrained_pars[i] = in__.scalar_lb_constrain(lower_bound[i]);
    } else {
      constrained_pars[i] = in__.scalar_lub_constrain(lower_bound[i],upper_bound[i]);
    }
  }
  for (auto & val : constrained_pars)
    cout << val << " ";
  cout << "\n\n";

  double logp_plus = my_8schools_model.log_prob<false, true, double>(cont_vector, params_i, msgs);

  cout << "log p = " << logp_plus << "\n";
  ////////////////////////
  // Check if we can get gradient
  ////////////////////////
  cout << "get gradient\n";
  stan::callbacks::interrupt interrupt;
  // Calculate simple finited difference gradient.
  stan::model::finite_diff_grad<false,true, model_8schools>
    (my_8schools_model, interrupt, unconstrained_params_r, params_i, gradient);

  for (auto & grad : gradient)
    cout << grad << " ";
  cout << "\n\n";

  double log_p_grad = stan::model::log_prob_grad<false, true, model_8schools>(my_8schools_model, unconstrained_params_r, params_i,gradient, msgs);
  cout << "gradient = " << log_p_grad << "\n\n";
  ////////////////////////
  // Do an optimisation
  ////////////////////////
  cout << "entering optimisation\n\n";
  std::vector<int> disc_vector;
  std::stringstream out;

  stan::optimization::ModelAdaptor<model_8schools> _adaptor(my_8schools_model, disc_vector, &out);
  cout << "Build model adaptor \n\n";
  stan::optimization::BFGSLineSearch<model_8schools,stan::optimization::LBFGSUpdate<> > lbfgs(my_8schools_model, cont_vector, params_i, &out);
  lbfgs._conv_opts.tolRelGrad =  1e+7;

  cout << "about to check step()\n\n";
  int ret = 0;
  while (ret == 0) {
    ret = lbfgs.step();
  }
  cout << lbfgs.get_code_string(ret) << "\n";
  // Print message
  cout << "ret = " << ret << endl;

  cout << "grad evals = " << lbfgs.grad_evals() << endl;
  cout << "logp = " << lbfgs.logp() << endl;
  cout << "grad norm = " << lbfgs.grad_norm() << endl;

  cout << "result = \n\n";
  cout << "size of current x = " << lbfgs.curr_x().size() << "\n";
  auto current_f =  lbfgs.curr_x();
  for (unsigned i = 0; i < current_f.size(); ++i) {
    std::cout << current_f[i] << " ";
  }
  vector<double> check_results;
  lbfgs.params_r(check_results);
  cout << "\n";

  stan::io::reader<double> in1__(check_results, params_i);
  cout << "re-transformed variables are" << endl;
  for (int i = 0; i < check_results.size(); ++i) {
    if ((lower_bound[i] <= -INF) & (upper_bound[i] >= INF)) {
      constrained_pars[i] = in1__.scalar_constrain();
    } else if (lower_bound[i] <= -INF & upper_bound[i] <= INF) {
      constrained_pars[i] = in1__.scalar_ub_constrain(upper_bound[i]);
    } else if (lower_bound[i] >= -INF & upper_bound[i] >= INF) {
      constrained_pars[i] = in1__.scalar_lb_constrain(lower_bound[i]);
    } else {
      constrained_pars[i] = in1__.scalar_lub_constrain(lower_bound[i],upper_bound[i]);
    }
  }

  for (auto & val : constrained_pars)
    cout << val << " ";
  cout << "\n\n";

  for(auto val:check_results)
    cout << val << " ";
  cout << "\n";

  ////////////////////////
  // Do an MCMC
  ////////////////////////
  // Mimick unit test
  /*
  * Runs HMC with NUTS with unit Euclidean
  * metric without adaptation.
  * following services/samples/hmc_nuts_unit_e.hpp
  */

  stan::mcmc::unit_e_nuts<model_8schools, boost::ecuyer1988> sampler(my_8schools_model, base_rng__);

  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0.0);
  sampler.set_max_depth(8);
  int num_warmup = 100;
  int num_samples = 200;
  int num_thin = 1;
  int refresh = 0;
  bool save_warmup = true;

  // These are easier to use compared to the actual
  stan::test::unit::instrumented_interrupt interrupt_test;
  stan::test::unit::instrumented_logger logger;
  stan::test::unit::instrumented_writer init, parameter, diagnostic;

  for (int i = 0; i < cont_vector.size(); ++i) {
    std::cout << cont_vector[i] << " ";
  }
  cout << "\n";

/*
  stan::services::util::run_sampler(sampler, my_8schools_model, cont_vector, num_warmup, num_samples,
                    num_thin, refresh, save_warmup, base_rng__, interrupt_test, logger,
                    parameter, diagnostic);

  cout << "number of iterations = " << interrupt_test.call_count() << endl;
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();
  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();

  // print some diagnostics
  cout << "number of rows " << parameter_names[0].size() << "\n";
  cout << "number of cols " << parameter_names.size() << "\n";

  for (int i = 0; i < parameter_names[0].size(); ++i) {
    std::cout << parameter_names[0][i] << " ";
  }
  cout << "\n";
  cout << "number of rows " << parameter_values.size() << " ncols = " << parameter_values[0].size() << "\n";
  cout << "number of rows " << parameter_names.size() << "\n";

  std::ofstream outFile("samples.txt");
  // the important part

  for (int i = 0; i < parameter_names[0].size(); ++i)
    outFile << parameter_names[0][i] << " ";
  outFile  << "\n";
  for (unsigned i = 0; i < parameter_values.size(); ++i) {
    for (unsigned j = 0; j < parameter_values[i].size(); ++j)
      outFile << parameter_values[i][j] << " ";
    outFile  << "\n";
  }

*/

  cout << "\n\n\n----------------------------------------------\n";
  cout << "--------------Linear regression section--------\n";
  cout << "----------------------------------------------\n";

  vector<Double> X = {8.8,15.4,51.2,21.4,22.6,54.3,29.2,5.3,6.3,11.1,44.5,27.2,28,22.2,8.9,55.7,30,19.3,34,10.5,1.4,15.6,0.5,5.4,7.5,13.7,36.8,23.1,2.8,45.1,28.5,14.1,37.9,37.6,36.4,33.8,31.1,18.8,13.9,12.4,6.1,15.8,5.3,63.4,44.2,2.5,11.9,10.7,35.6,18.3};
  vector<Double> Y = {23.67,36.05,119.67,65.28,50.75,143.87,51.82,19.08,16.41,28.65,108.58,58.51,62.22,41.2,9.5,133.96,75.4,46.06,89.91,47.18,-1.74,11.73,12.43,5.25,10.4,43.53,83.34,41.11,8.85,104.35,67.08,37.51,84.98,95.31,83.1,83.02,85.02,49.05,29.31,41.83,25.4,43.26,15.32,141.85,118.54,-0.35,51.95,42.02,81.07,31.99};
  Double linear_sigma = 10.87;

  model_linear* my_linear_model = new model_linear(&std::cout, random_seed, Y, X, linear_sigma);
  cout << "succefully constructed my linear regression class\n";
  // use MLE estimates from Stan
  params_r = {0.324,  1.34};
  vector<Double> params_r_;
  for(unsigned i = 0; i < params_r.size(); ++i) {
    Double var_i(params_r[i]);
    cout << "val = " << var_i << " ";
    params_r_.push_back(var_i);
  }

  cont_vector.resize(params_r.size(),0.0);
  cout << "\nStarting values for a and b = " << params_r[0] << " & " << params_r[1] << "\n";

  Double val = my_linear_model->log_prob<true,false,Double>(params_r_, params_i, msgs);

  val.grad();
  stan::math::print_stack(std::cout);

  for (unsigned i = 0; i < params_r_.size(); ++i) {
    std::cout << "val " << params_r_[i].val() << " adj = " << params_r_[i].adj() << endl;
  }

  cout << "val = " << val.val() << " should be -362.166 \n";
  log_p_grad = stan::model::log_prob_grad<false, true, model_linear>((*my_linear_model), params_r, params_i, gradient, msgs);
  cout << "log_p_grad " << log_p_grad << " should be -362.166 \n";
  cout << "gradient by value \n";
  vector<double> actual_grad = {10.2889, 337.994};
  for (unsigned i = 0; i < gradient.size(); ++i) {
    std::cout << gradient[i] << " (" << actual_grad[i] << ") ";
  }
  cout << "\n";


  stan::optimization::BFGSLineSearch<model_linear,stan::optimization::LBFGSUpdate<> > linear_lbfgs((*my_linear_model), params_r, params_i, &out);
  linear_lbfgs._conv_opts.tolRelGrad =  1e+7;


  cout << "about to check step()\n\n";
  ret = 0;
  while (ret == 0) {
    ret = linear_lbfgs.step();
  }
  cout << linear_lbfgs.get_code_string(ret) << "\n";
  // Print message
  cout << "ret = " << ret << endl;

  cout << "grad evals = " << linear_lbfgs.grad_evals() << " should be 8" << endl;
  cout << "logp = " << linear_lbfgs.logp() << " should be -185.332"<< endl;
  cout << "grad norm = " << linear_lbfgs.grad_norm() << endl;

  cout << "result = \n\n";
  cout << "size of current x = " << linear_lbfgs.curr_x().size() << "\n";
  current_f =  linear_lbfgs.curr_x();
  for (unsigned i = 0; i < current_f.size(); ++i) {
    std::cout << current_f[i] << " ";
  }
  cout << "\nWhen fitting answer in R we get  2.069 2.333\n";



  // Test map utility with var class
  map<unsigned, Double> test_map_;
  Double x1 = 3234.2;
  unsigned start_year = 1990;
  cout << test_map_.size() << "\n";
  test_map_[start_year] = x1;
  cout << test_map_.size() << " val = " << test_map_[start_year] <<  "\n";


  // Test passing values by reference
  cout << "value by reference " << endl;
  Double x2;
  auto& ref_x2 = x2;

  ref_x2 = 1.0;
  cout << ref_x2 << "\n";





  ////////---------------///////////////
  // The var.hpp class doesn't have a iostream << operator, so I am learning how to overload
  // we need this in Casal2 because we use boosts lexical casts in the Utilities\To.h file
  // I was reading that this needs both << and >> operators.
  /*
  Double x1 = 3234.2;
  cout << "x1 = " << x1 << endl;

  Double x2;
  cout << "is uninitialised = " << x2.is_uninitialized() << "\n";
  x2 = 3234.2;
  cout << "x2 = " << x2 << endl;
  cout << "is uninitialised = " << x2.is_uninitialized() << "\n";

  // change the number
  x2 = 23.2;
  cout << "x2 is now = " << x2 << endl;


  x2 += 23;

  cout << "x2 = " << x2 << endl;

  cout << "assign a string\n";

  std::string str = "12.4";

  cout << str << "\n";

  std::istringstream iss (str);

  double val;
  iss >> val;
  cout << "val " << val << "\n";

  cout << "cannot assing...\n";

  Double x3;
  cout << "is uninitialised = " << x3.is_uninitialized() << "\n";
  cout << "The complex object is ";
  x3 = val;
  cout <<"\n"<< x3 << "\n";

  x3+= 10;
  cout <<"\n"<< x3 << "\n";



  TemplateTest my_test;
  double test_val = 213.23;
  double test_result = my_test.test(test_val);
  cout << test_result << "\n";



  AltTemplateTest alt_my_test;
  var alt_test_val(213.23);
  double alt_test_result = alt_my_test.test<var>(alt_test_val);
  cout << stan::math::value_of(alt_test_result) << "\n";
*/

  std::cout << "\ncomplete";
	return 0;
}
