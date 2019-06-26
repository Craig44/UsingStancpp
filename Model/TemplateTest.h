/*
 * TemplateTest.h
 *
 *  Created on: 26/06/2019
 *      Author: Dell User
 */

#ifndef MODEL_TEMPLATETEST_H_
#define MODEL_TEMPLATETEST_H_

namespace niwa {
  class TemplateTest {
  public:
    TemplateTest();
    virtual ~TemplateTest();
    template<typename T_>
    T_                        test(T_ val) const;

  }; //TemplateTest
} //niwa namespace
#endif /* MODEL_TEMPLATETEST_H_ */
