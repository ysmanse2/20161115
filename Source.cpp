#include <iostream>

// linear regression < deep learning < machine learning
class LinearHypothesis
{
public:
	// linear hypothesis : y = a * x + b
	float a_ = 0.0f;
	float b_ = 0.0f;

	float a__ = 0.0f;
	float b__ = 0.0f;

	float getY(const float& x_input, const float& y_target)
	{
		float y_ = a_ * x_input + b_;
		float error = y_ - y_target;

		float sqr_error = 0.5 * error * error;

		float dse_over_da = error * x_input;
		float dse_over_db = error;

		float lr = 0.1; // small number
		a_ -= dse_over_da * lr;
		b_ -= dse_over_db * lr;


		return a__ * y_ + b__; // returns y = a*x+b
	}

	float getY_(const float& x_input)
	{
		return a__ * (a_ * x_input + b_) + b__ ; // returns y = a*x+b
	}
};

const int num_data = 5;

int main()
{
	// 0 hour -> 0 pts
	// 1 hour -> 2 pts
	// 2 hour -> 4 pts
	// 2.5 hour -> ? (human can do this. and let machine do this.)
	// 3 hour -> ?

	const float study_time_data[num_data] = { 0.1, 0.2, 0.3, 0.4, 0.5 };
	const float score_data[num_data] = { 4, 5, 6, 7, 8 };

	// input x is study time -> black box(AI) -> output y is score
	// linear hypothesis : y = a * x + b
	LinearHypothesis lh;

	for (int tr = 0; tr < 1000; tr++)
		for (int i = 0; i < num_data; i++)
		{
			// let's train our linear hypothesis to answer correctly!
			const float x_input = study_time_data[i];
			const float y_output = lh.getY(x_input, score_data[i]);
			const float y_target = score_data[i];
			const float error = y_output - y_target;
			// we can consider that our LH is trained well when error is 0 or small enough
			// we define squared error
			const float sqr_error = 0.5 * error * error; // always zero or positive

														 // we want to find good combination of a and b which minimizes sqr_error

														 // sqr_error = 0.5 * (a * x + b - y_target)^2
														 // d sqr_error / da = 2*0.5*(a * x + b - y_target) * x; 
														 // d sqr_error / db = 2*0.5*(a * x + b - y_target) * 1;

			const float dse_over_da = error * x_input;
			const float dse_over_db = error;

			// need to find good a and b
			// we can update a and b to decrease squared error
			// this is the gradient descent method
			// learning rate
			const float lr = 0.1; // small number
			lh.a__ -= dse_over_da * lr;
			lh.b__ -= dse_over_db * lr;
		}

	

	// trained hypothesis
	std::cout << "input 0.1" << std::endl;
	std::cout << "From Doubly layered trained hypothesis " << lh.getY_(0.1) << std::endl;
	std::cout << "input 0.2" << std::endl;
	std::cout << "From Doubly layered trained hypothesis " << lh.getY_(0.2) << std::endl;
	std::cout << "input 0.3" << std::endl;
	std::cout << "From Doubly layered trained hypothesis " << lh.getY_(0.3) << std::endl;
	std::cout << "input 0.4" << std::endl;
	std::cout << "From Doubly layered trained hypothesis " << lh.getY_(0.4) << std::endl;
	std::cout << "input 0.5" << std::endl;
	std::cout << "From Doubly layered trained hypothesis " << lh.getY_(0.5) << std::endl;
	std::cout << "input 10.5" << std::endl;
	std::cout << "From Doubly layered trained hypothesis " << lh.getY_(10.5) << std::endl;
	

	return 0;
}