#include "fparser.hh"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    unsigned callback_calls = 0;
    unsigned sequence_value = 0;

    double count(const double* values)
    {
        ++callback_calls;
        return values[0];
    }

    double sequence(const double*)
    {
        ++callback_calls;
        return double(++sequence_value);
    }

    void evaluate(FunctionParser& parser,
                  const char* label,
                  const std::string& expression,
                  const std::string& variables,
                  const std::vector<double>& values,
                  bool use_degrees)
    {
        const int parse_result = parser.Parse(expression, variables, use_degrees);
        if(parse_result >= 0)
        {
            std::cout << label << "|parse|" << parse_result << "|"
                      << int(parser.GetParseErrorType()) << "\n";
            return;
        }
        const double* variable_values = values.empty() ? 0 : &values[0];
        callback_calls = 0;
        sequence_value = 0;
        const double result = parser.Eval(variable_values);
        std::cout << label << "|eval|" << std::setprecision(17) << result
                  << "|" << parser.EvalError() << "\n";
    }

    int evaluate_corpus(const char* path)
    {
        std::ifstream input(path);
        if(!input)
        {
            std::cerr << "could not open differential corpus: " << path << "\n";
            return 2;
        }

        std::string line;
        unsigned line_number = 0;
        while(std::getline(input, line))
        {
            ++line_number;
            if(line.empty() || line[0] == '#') continue;

            std::vector<std::string> fields;
            std::istringstream line_stream(line);
            std::string field;
            while(std::getline(line_stream, field, '\t'))
                fields.push_back(field);
            if(fields.size() != 5)
            {
                std::cerr << "invalid corpus row " << line_number << "\n";
                return 2;
            }

            FunctionParser square;
            square.Parse("x^2", "x");
            FunctionParser sum;
            sum.Parse("a+b", "a,b");
            FunctionParser negative_root;
            negative_root.Parse("sqrt(-x)", "x");
            FunctionParser parser;
            parser.AddConstant("pi", 3.1415926535897932);
            parser.AddUnit("cm", 0.01);
            parser.AddFunction("count", &count, 1);
            parser.AddFunction("sequence", &sequence, 1);
            parser.AddFunction("square", square);
            parser.AddFunction("sum2", sum);
            parser.AddFunction("root", negative_root);
            if(fields[2] == "@deduce")
            {
                std::vector<std::string> deduced;
                const int parse_result =
                    parser.ParseAndDeduceVariables(fields[1], deduced,
                                                   fields[3] == "1");
                if(parse_result >= 0)
                {
                    std::cout << fields[0] << "\tparse\t" << parse_result
                              << "\t" << int(parser.GetParseErrorType()) << "\n";
                    continue;
                }
                std::string variable_list;
                for(std::vector<std::string>::const_iterator iter =
                        deduced.begin();
                    iter != deduced.end(); ++iter)
                {
                    if(iter != deduced.begin()) variable_list += ",";
                    variable_list += *iter;
                }
                std::cout << fields[0] << "\tvars\t" << variable_list
                          << "\t-1\n";
                continue;
            }
            std::string variables = fields[2];
            const bool optimize = variables.compare(0, 10, "@optimize:") == 0;
            if(optimize) variables = variables.substr(10);
            const int parse_result =
                parser.Parse(fields[1], variables, fields[3] == "1");
            if(parse_result >= 0)
            {
                std::cout << fields[0] << "\tparse\t" << parse_result << "\t"
                          << int(parser.GetParseErrorType()) << "\n";
                continue;
            }
            if(optimize) parser.Optimize();

            std::vector<double> values;
            if(fields[4] != "-")
            {
                std::istringstream value_stream(fields[4]);
                std::string value;
                while(std::getline(value_stream, value, ','))
                    values.push_back(std::strtod(value.c_str(), 0));
            }
            const double* variable_values = values.empty() ? 0 : &values[0];
            callback_calls = 0;
            sequence_value = 0;
            const double result = parser.Eval(variable_values);
            std::cout << fields[0] << "\teval\t" << std::setprecision(17)
                      << result << "\t" << parser.EvalError() << "\n";
            std::cout << fields[0] << ".calls\tcalls\t" << callback_calls
                            << "\t0\n";
        }
        return 0;
    }
}

int main(int argc, char** argv)
{
    if(argc == 2) return evaluate_corpus(argv[1]);

    FunctionParser parser;
    parser.AddConstant("pi", 3.1415926535897932);
    parser.AddUnit("cm", 0.01);
    parser.AddFunction("count", &count, 1);
    parser.AddFunction("sequence", &sequence, 1);

    evaluate(parser, "unary-power", "-2^2", "", std::vector<double>(), false);
    evaluate(parser, "right-power", "2^3^2", "", std::vector<double>(), false);
    evaluate(parser, "negative-power", "2^-2", "", std::vector<double>(), false);
    evaluate(parser, "degree-sin", "sin(30)", "", std::vector<double>(), true);
    evaluate(parser, "degree-asin", "asin(0.5)", "", std::vector<double>(), true);
    evaluate(parser, "unit", "2cm", "", std::vector<double>(), false);
    std::vector<double> epsilon_values(2);
    epsilon_values[0] = 1.0;
    epsilon_values[1] = 1.0 + 5.0e-13;
    evaluate(parser, "epsilon-equal", "x=y", "x,y", epsilon_values, false);
    std::vector<double> one_value(1);
    one_value[0] = 0.49;
    evaluate(parser, "truth-not", "!x", "x", one_value, false);
    one_value[0] = 0.5;
    evaluate(parser, "truth-double-not", "!!x", "x", one_value, false);

    evaluate(parser, "divide-zero", "1/0", "", std::vector<double>(), false);
    evaluate(parser, "sqrt-negative", "sqrt(-1)", "", std::vector<double>(), false);
    evaluate(parser, "log-zero", "log(0)", "", std::vector<double>(), false);
    evaluate(parser, "asin-domain", "asin(2)", "", std::vector<double>(), false);
    evaluate(parser, "zero-negative-power", "0^-1", "", std::vector<double>(), false);
    evaluate(parser, "pow-zero-negative", "pow(0,-1)", "", std::vector<double>(), false);
    evaluate(parser, "zero-fractional-power", "0^-0.5", "", std::vector<double>(), false);
    std::vector<double> power_values(2);
    power_values[0] = 0.0;
    power_values[1] = -1.0;
    evaluate(parser, "dynamic-zero-power", "x^y", "x,y", power_values, false);
    evaluate(parser, "dynamic-zero-pow", "pow(x,y)", "x,y", power_values, false);
    evaluate(parser, "constant-exponent-zero-base", "pow(x,-1)", "x",
             std::vector<double>(1, 0.0), false);

    callback_calls = 0;
    evaluate(parser, "lazy-if", "if(0,count(x),7)", "x",
             one_value, false);
    evaluate(parser, "lazy-and", "0&count(x)", "x",
             one_value, false);
    evaluate(parser, "lazy-or", "1|count(x)", "x",
             one_value, false);
    std::cout << "lazy-calls|count|" << callback_calls << "\n";

    callback_calls = 0;
    one_value[0] = 2.0;
    evaluate(parser, "inline", "a:=count(x); b:=a+1; b+b", "x",
             one_value, false);
    std::cout << "inline-calls|count|" << callback_calls << "\n";

    const char* builtin_cases[] =
    {
        "abs(-3)", "acos(1)", "acosh(1)", "asin(0)", "asinh(0)",
        "atan(0)", "atan2(0,1)", "atanh(0)", "cbrt(-8)", "ceil(1.2)",
        "cos(0)", "cosh(0)", "cot(pi/4)", "csc(pi/2)", "exp(0)",
        "exp2(3)", "floor(1.8)", "hypot(3,4)", "if(0,1,2)", "int(-1.5)",
        "log(1)", "log10(100)", "log2(8)", "max(1,2)", "min(1,2)",
        "pow(2,3)", "sec(0)", "sin(pi/2)", "sinh(0)", "sqrt(4)",
        "tan(0)", "tanh(0)", "trunc(-1.8)"
    };
    const unsigned builtin_count =
        sizeof(builtin_cases) / sizeof(builtin_cases[0]);
    for(unsigned i = 0; i < builtin_count; ++i)
        evaluate(parser, builtin_cases[i], builtin_cases[i], "",
                 std::vector<double>(), false);
    return 0;
}
