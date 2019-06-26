/*
 * AltTemplateTest.hpp
 *
 *  Created on: 26/06/2019
 *      Author: Dell User
 */

#ifndef MODEL_ALTTEMPLATETEST_H_
#define MODEL_ALTTEMPLATETEST_H_

namespace niwa {
  class AltTemplateTest {
  public:
    AltTemplateTest() { };
    virtual ~AltTemplateTest() { };
    template<typename T_>
    T_ test(T_ val)     const {
      T_ return_val = val;
    return return_val;
  };

  }; //AltTemplateTest
} //niwa namespace
#endif /* MODEL_ALTTEMPLATETEST_H_ */
