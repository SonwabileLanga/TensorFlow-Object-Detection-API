# Description:
# Python support for TensorFlow.
#
# Public targets:
#  ":platform" - Low-level and platform-specific Python code.

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//tensorflow:internal",
    "//tensorflow/lite/toco/python:__pkg__",
    "//tensorflow_models:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/py/tensorflow_examples:__subpackages__",
    "//third_party/py/tf_slim:__subpackages__",
    # TODO(aselle): to pass open source test.
    "//bazel_pip/tensorflow/lite/toco/python:__pkg__",
]

load("//tensorflow:tensorflow.bzl", "if_not_v2", "if_not_windows", "tf_cuda_library", "tf_gen_op_wrapper_py", "py_test", "tf_py_test", "py_tests", "tf_py_build_info_genrule", "tf_cc_shared_object", "tf_custom_op_library_additional_deps_impl")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow/core:platform/default/build_config.bzl", "pyx_library", "tf_proto_library", "tf_proto_library_py", "tf_additional_lib_deps", "tf_additional_all_protos", "tf_protos_grappler")
load("//tensorflow/core:platform/default/build_config_root.bzl", "tf_additional_plugin_deps", "tf_additional_verbs_deps", "tf_additional_mpi_deps", "tf_additional_gdr_deps", "if_static")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")
load(
    "//third_party/ngraph:build_defs.bzl",
    "if_ngraph",
)

package(
    default_visibility = visibility,
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

exports_files(["platform/base.i"])

py_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/aot/tests:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/contrib/learn:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/contrib/learn/python/learn/datasets:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/lite/toco/python:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/debug:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/tools:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/tools/quantization:__pkg__",  # TODO(b/34059704): remove when fixed
    ],
    deps = [
        ":no_contrib",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/tpu:tpu_estimator",
    ] + if_not_v2(["//tensorflow/contrib:contrib_py"]),
)

py_library(
    name = "keras_lib",
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:internal",
        "//tensorflow/python/estimator:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//tensorflow/tools/api/tests:__pkg__",
        "//tensorflow/tools/compatibility/update:__pkg__",
        "//tensorflow_estimator:__subpackages__",
    ],
    deps = [
        ":rnn",
        "//tensorflow/python:layers",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/keras",
    ],
)

py_library(
    name = "no_contrib",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python/estimator:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//tensorflow/tools/api/tests:__pkg__",
        "//tensorflow/tools/compatibility/update:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
    ],
    deps = [
        ":array_ops",
        ":audio_ops_gen",
        ":bitwise_ops",
        ":boosted_trees_ops",
        ":check_ops",
        ":client",
        ":client_testlib",
        ":clustering_ops",
        ":collective_ops",
        ":cond_v2",
        ":config",
        ":confusion_matrix",
        ":control_flow_ops",
        ":cudnn_rnn_ops_gen",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":graph_util",
        ":histogram_ops",
        ":image_ops",
        ":initializers_ns",
        ":io_ops",
        ":keras_lib",
        ":kernels",
        ":lib",
        ":list_ops",
        ":manip_ops",
        ":map_fn",
        ":math_ops",
        ":metrics",
        ":nccl_ops",
        ":nn",
        ":ops",
        ":platform",
        ":proto_ops",
        ":pywrap_tensorflow",
        ":rnn_ops_gen",
        ":saver_test_utils",
        ":script_ops",
        ":session_ops",
        ":sets",
        ":sparse_ops",
        ":spectral_ops_test_util",
        ":standard_ops",
        ":state_ops",
        ":string_ops",
        ":subscribe",
        ":summary",
        ":tensor_array_ops",
        ":tensor_forest_ops",
        ":test_ops",  # TODO: Break testing code out into separate rule.
        ":tf_cluster",
        ":tf_item",
        ":tf_optimizer",
        ":training",
        ":util",
        ":weights_broadcast_ops",
        ":while_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/compat",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/compiler",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:estimator_training",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:execution_callbacks",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:profiler",
        "//tensorflow/python/eager:profiler_client",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/parallel_for",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python/profiler",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/tools:module_util",
        "//tensorflow/python/tools/api/generator:create_python_api",
        "//tensorflow/python/tpu:tpu_noestimator",
        "//third_party/py/numpy",
    ],
)

tf_py_build_info_genrule()

py_library(
    name = "platform",
    srcs = glob(
        [
            "platform/*.py",
        ],
        exclude = [
            "**/*test.py",
            "**/benchmark.py",  # In platform_benchmark.
        ],
    ) + ["platform/build_info.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lib",
        ":pywrap_tensorflow",
        ":util",
        "//tensorflow/core:protos_all_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

py_library(
    name = "platform_benchmark",
    srcs = ["platform/benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":platform",
        "@six_archive//:six",
    ],
)

py_library(
    name = "platform_test",
    srcs = ["platform/googletest.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":platform_benchmark",
        "@absl_py//absl/testing:absltest",
    ],
)

tf_py_test(
    name = "resource_loader_test",
    size = "small",
    srcs = ["platform/resource_loader_test.py"],
    additional_deps = [
        ":platform",
        ":platform_test",
    ],
)

tf_py_test(
    name = "flags_test",
    size = "small",
    srcs = ["platform/flags_test.py"],
    additional_deps = [
        ":client_testlib",
        ":platform",
    ],
)

tf_py_test(
    name = "stacktrace_handler_test",
    size = "small",
    srcs = ["platform/stacktrace_handler_test.py"],
    additional_deps = [
        ":client_testlib",
        ":platform",
    ],
    tags = [
        "no_windows",
        "nomac",
    ],
)

tf_py_test(
    name = "app_test",
    size = "small",
    srcs = ["platform/app_test.py"],
    additional_deps = [":platform"],
    tags = ["notap"],
)

cc_library(
    name = "cost_analyzer_lib",
    srcs = ["grappler/cost_analyzer.cc"],
    hdrs = ["grappler/cost_analyzer.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:analytical_cost_estimator",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/grappler/costs:measuring_cost_estimator",
        "//tensorflow/core/grappler/costs:utils",
    ] + tf_protos_grappler(),
)

cc_library(
    name = "model_analyzer_lib",
    srcs = ["grappler/model_analyzer.cc"],
    hdrs = ["grappler/model_analyzer.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/costs:graph_properties",
    ],
)

cc_library(
    name = "numpy_lib",
    srcs = ["lib/core/numpy.cc"],
    hdrs = ["lib/core/numpy.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "bfloat16_lib",
    srcs = ["lib/core/bfloat16.cc"],
    hdrs = ["lib/core/bfloat16.h"],
    deps = [
        ":numpy_lib",
        ":safe_ptr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "ndarray_tensor_bridge",
    srcs = ["lib/core/ndarray_tensor_bridge.cc"],
    hdrs = ["lib/core/ndarray_tensor_bridge.h"],
    visibility = visibility + [
        "//learning/deepmind/courier:__subpackages__",
    ],
    deps = [
        ":bfloat16_lib",
        ":numpy_lib",
        "//tensorflow/c:c_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "py_exception_registry",
    srcs = ["lib/core/py_exception_registry.cc"],
    hdrs = ["lib/core/py_exception_registry.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "kernel_registry",
    srcs = ["util/kernel_registry.cc"],
    hdrs = ["util/kernel_registry.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "cpp_python_util",
    srcs = ["util/util.cc"],
    hdrs = ["util/util.h"],
    deps = [
        ":safe_ptr",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "py_func_lib",
    srcs = ["lib/core/py_func.cc"],
    hdrs = ["lib/core/py_func.h"],
    deps = [
        ":ndarray_tensor_bridge",
        ":numpy_lib",
        ":py_util",
        ":safe_ptr",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:script_ops_op_lib",
        "//tensorflow/python:ndarray_tensor",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "safe_ptr",
    srcs = ["lib/core/safe_ptr.cc"],
    hdrs = ["lib/core/safe_ptr.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "ndarray_tensor",
    srcs = ["lib/core/ndarray_tensor.cc"],
    hdrs = ["lib/core/ndarray_tensor.h"],
    visibility = visibility + [
        "//learning/deepmind/courier:__subpackages__",
    ],
    deps = [
        ":bfloat16_lib",
        ":ndarray_tensor_bridge",
        ":numpy_lib",
        ":safe_ptr",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "py_seq_tensor",
    srcs = ["lib/core/py_seq_tensor.cc"],
    hdrs = ["lib/core/py_seq_tensor.h"],
    deps = [
        ":numpy_lib",
        ":py_util",
        ":safe_ptr",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "py_util",
    srcs = ["lib/core/py_util.cc"],
    hdrs = ["lib/core/py_util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:script_ops_op_lib",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "py_record_reader_lib",
    srcs = ["lib/io/py_record_reader.cc"],
    hdrs = ["lib/io/py_record_reader.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "py_record_writer_lib",
    srcs = ["lib/io/py_record_writer.cc"],
    hdrs = ["lib/io/py_record_writer.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_shared_object(
    name = "framework/test_file_system.so",
    srcs = ["framework/test_file_system.cc"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    linkopts = select({
        "//conditions:default": [
            "-lm",
        ],
        "//tensorflow:macos": [],
        "//tensorflow:windows": [],
    }),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

tf_py_test(
    name = "file_system_test",
    size = "small",
    srcs = ["framework/file_system_test.py"],
    additional_deps = [
        ":client_testlib",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":platform",
        ":util",
    ],
    data = [":framework/test_file_system.so"],
    main = "framework/file_system_test.py",
    tags = [
        "no_pip",  # Path issues due to test environment
        "no_windows",
        "notap",
    ],
)

tf_py_test(
    name = "decorator_utils_test",
    srcs = ["util/decorator_utils_test.py"],
    additional_deps = [
        ":client_testlib",
        ":platform",
        ":util",
    ],
)

tf_py_test(
    name = "tf_export_test",
    srcs = ["util/tf_export_test.py"],
    additional_deps = [
        ":client_testlib",
        ":platform",
        ":util",
    ],
)

tf_py_test(
    name = "deprecation_test",
    srcs = ["util/deprecation_test.py"],
    additional_deps = [
        ":client_testlib",
        ":platform",
        ":util",
    ],
)

tf_py_test(
    name = "dispatch_test",
    srcs = ["util/dispatch_test.py"],
    additional_deps = [
        ":client_testlib",
        ":platform",
        ":util",
    ],
)

tf_py_test(
    name = "keyword_args_test",
    srcs = ["util/keyword_args_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = [
        "framework/python_op_gen.cc",
        "framework/python_op_gen_internal.cc",
    ],
    hdrs = [
        "framework/python_op_gen.h",
        "framework/python_op_gen_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "python_op_gen_main",
    srcs = ["framework/python_op_gen_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":python_op_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

py_library(
    name = "framework_for_generated_wrappers",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":constant_op",
        ":device",
        ":device_spec",
        ":dtypes",
        ":framework_ops",
        ":function",
        ":op_def_library",
        ":op_def_registry",
        ":registry",
        ":tensor_shape",
        ":versions",
    ],
)

# What is needed for tf_gen_op_wrapper_py. This is the same as
# "framework_for_generated_wrappers" minus the "function" dep. This is to avoid
# circular dependencies, as "function" uses generated op wrappers.
py_library(
    name = "framework_for_generated_wrappers_v2",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":constant_op",
        ":device",
        ":device_spec",
        ":dtypes",
        ":framework_ops",
        ":op_def_library",
        ":op_def_registry",
        ":registry",
        ":tensor_shape",
        ":versions",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "subscribe",
    srcs = ["framework/subscribe.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_ops",
        ":platform",
        ":variables",
    ],
)

py_library(
    name = "framework",
    srcs = [
        "framework/framework_lib.py",
        "framework/graph_io.py",
        "framework/importer.py",
        "framework/load_library.py",
        "framework/meta_graph.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":common_shapes",
        ":composite_tensor",
        ":convert_to_constants",
        ":cpp_shape_inference_proto_py",
        ":errors",
        ":framework_fast_tensor_util",
        ":framework_for_generated_wrappers",
        ":function",
        ":graph_util",
        ":lib",
        ":platform",
        ":pywrap_tensorflow",
        ":random_seed",
        ":sparse_tensor",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        ":util",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "c_api_util",
    srcs = ["framework/c_api_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "common_shapes",
    srcs = ["framework/common_shapes.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cpp_shape_inference_proto_py",
        ":errors",
        ":framework_ops",
        ":pywrap_tensorflow",
        ":tensor_shape",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "constant_op",
    srcs = ["framework/constant_op.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:execute",
    ],
)

py_library(
    name = "device_spec",
    srcs = ["framework/device_spec.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
    ],
)

py_library(
    name = "device",
    srcs = ["framework/device.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "dtypes",
    srcs = ["framework/dtypes.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "errors",
    srcs = [
        "framework/errors.py",
        "framework/errors_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":c_api_util",
        ":error_interpolation",
        ":util",
    ],
)

py_library(
    name = "error_interpolation",
    srcs = [
        "framework/error_interpolation.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
    ],
)

py_library(
    name = "function",
    srcs = ["framework/function.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":graph_to_function_def",
        ":op_def_registry",
        ":util",
        ":variable_scope",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "graph_to_function_def",
    srcs = ["framework/graph_to_function_def.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":op_def_registry",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "function_def_to_graph",
    srcs = ["framework/function_def_to_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_ops",
        ":function",
        ":tensor_shape",
        ":versions",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "function_def_to_graph_test",
    size = "small",
    srcs = ["framework/function_def_to_graph_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":constant_op",
        ":dtypes",
        ":framework_ops",
        ":function",
        ":function_def_to_graph",
        ":graph_to_function_def",
        ":math_ops",
        ":test_ops",
    ],
    tags = ["no_pip"],
)

py_library(
    name = "graph_util",
    srcs = [
        "framework/graph_util.py",
        "framework/graph_util_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "convert_to_constants",
    srcs = [
        "framework/convert_to_constants.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_util",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "kernels",
    srcs = [
        "framework/kernels.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "op_def_library",
    srcs = ["framework/op_def_library.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_shape",
        ":util",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "op_def_registry",
    srcs = ["framework/op_def_registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "framework_ops",  # "ops" is already the name of a deprecated target
    srcs = ["framework/ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":c_api_util",
        ":control_flow_util",
        ":device",
        ":dtypes",
        ":error_interpolation",
        ":indexed_slices",
        ":op_def_registry",
        ":platform",
        ":registry",
        ":tensor_conversion_registry",
        ":tensor_like",
        ":tensor_shape",
        ":tf2",
        ":traceable_stack",
        ":type_spec",
        ":util",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:tape",
        "@six_archive//:six",
    ],
)

py_library(
    name = "tensor_like",
    srcs = ["framework/tensor_like.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "indexed_slices",
    srcs = ["framework/indexed_slices.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":type_spec",
        ":util",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "tensor_conversion_registry",
    srcs = ["framework/tensor_conversion_registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "map_fn",
    srcs = ["ops/map_fn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":util",
        ":variable_scope",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "func_graph",
    srcs = ["framework/func_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":auto_control_deps",
        ":framework_ops",
        ":sparse_tensor",
        ":tensor_array_ops",
        "//tensorflow/python/autograph",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "auto_control_deps",
    srcs = ["framework/auto_control_deps.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":framework_ops",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":util",
    ],
)

tf_py_test(
    name = "auto_control_deps_test",
    size = "small",
    srcs = ["framework/auto_control_deps_test.py"],
    additional_deps = [
        ":auto_control_deps",
        ":client_testlib",
        "//tensorflow/python/keras",
    ],
)

py_library(
    name = "config",
    srcs = ["framework/config.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_ops",
        ":util",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "config_test",
    size = "small",
    srcs = ["framework/config_test.py"],
    additional_deps = [
        ":config",
        ":constant_op",
        ":client_testlib",
        ":platform",
        ":test_ops",
        ":util",
    ],
    tags = ["no_pip"],  # test_ops are not available in pip.
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "random_seed",
    srcs = ["framework/random_seed.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_ops",
    ],
)

py_library(
    name = "registry",
    srcs = ["framework/registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":platform",
        ":util",
    ],
)

py_library(
    name = "smart_cond",
    srcs = ["framework/smart_cond.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":tensor_util",
    ],
)

tf_py_test(
    name = "smart_cond_test",
    size = "small",
    srcs = ["framework/smart_cond_test.py"],
    additional_deps = [
        ":client_testlib",
        ":constant_op",
        ":framework_ops",
        ":math_ops",
        ":session",
        ":smart_cond",
    ],
)

py_library(
    name = "sparse_tensor",
    srcs = ["framework/sparse_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":framework_ops",
        ":tensor_like",
        ":tensor_util",
        ":type_spec",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["framework/composite_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":tensor_util",
    ],
)

py_library(
    name = "composite_tensor_utils",
    srcs = ["framework/composite_tensor_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":composite_tensor",
        ":sparse_ops",
        ":sparse_tensor",
        "//tensorflow/python/ops/ragged:ragged_concat_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "framework_composite_tensor_test",
    srcs = ["framework/composite_tensor_test.py"],
    main = "framework/composite_tensor_test.py",
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":composite_tensor",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "framework_composite_tensor_utils_test",
    srcs = ["framework/composite_tensor_utils_test.py"],
    additional_deps = [
        ":array_ops",
        ":composite_tensor",
        ":composite_tensor_utils",
        ":framework_test_lib",
        ":sparse_ops",
        ":sparse_tensor",
        "//third_party/py/numpy",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
    ],
    main = "framework/composite_tensor_utils_test.py",
)

# This target is maintained separately from :util to provide separate visibility
# for legacy users who were granted visibility when the functions were private
# members of ops.Graph.
py_library(
    name = "tf_stack",
    srcs = ["util/tf_stack.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "tensor_shape",
    srcs = ["framework/tensor_shape.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":tf2",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:monitoring",
    ],
)

py_library(
    name = "type_spec",
    srcs = ["framework/type_spec.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":tensor_shape",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tensor_spec",
    srcs = ["framework/tensor_spec.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common_shapes",
        ":dtypes",
        ":tensor_shape",
        ":type_spec",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tensor_util",
    srcs = ["framework/tensor_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_like",
        ":tensor_shape",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "traceable_stack",
    srcs = ["framework/traceable_stack.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":util",
    ],
)

py_library(
    name = "versions",
    srcs = ["framework/versions.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
    ],
)

py_library(
    name = "extra_py_tests_deps",
    srcs_version = "PY2AND3",
    deps = [
        ":keras_lib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "framework_test_lib",
    srcs = ["framework/test_util.py"],
    srcs_version = "PY2AND3",
    visibility = visibility + [
        "//tensorflow_estimator/python/estimator:__subpackages__",
    ],
    deps = [
        ":array_ops",
        ":client",
        ":errors",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":random_seed",
        ":resource_variable_ops",
        ":session",
        ":tensor_array_ops",
        ":training",
        ":util",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

# Including this as a dependency will result in tests using
# :framework_test_lib to use XLA.
py_library(
    name = "is_xla_test_true",
    srcs = ["framework/is_xla_test_true.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "distributed_framework_test_lib",
    srcs_version = "PY2AND3",
    deps = [":framework_test_lib"],
)

py_library(
    name = "client_testlib",
    srcs = ["platform/test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":cond_v2",
        ":framework_test_lib",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":platform_test",
        ":util",
        ":while_v2",
    ],
)

tf_py_test(
    name = "framework_registry_test",
    size = "small",
    srcs = ["framework/registry_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    main = "framework/registry_test.py",
)

tf_py_test(
    name = "framework_errors_test",
    size = "small",
    srcs = ["framework/errors_test.py"],
    additional_deps = [
        ":client_testlib",
        ":errors",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/errors_test.py",
)

tf_py_test(
    name = "framework_error_interpolation_test",
    size = "small",
    srcs = ["framework/error_interpolation_test.py"],
    additional_deps = [
        ":client_testlib",
        ":constant_op",
        ":error_interpolation",
        ":traceable_stack",
    ],
    main = "framework/error_interpolation_test.py",
)

tf_py_test(
    name = "framework_subscribe_test",
    size = "small",
    srcs = ["framework/subscribe_test.py"],
    additional_deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":script_ops",
        ":subscribe",
    ],
    main = "framework/subscribe_test.py",
)

tf_py_test(
    name = "contrib_test",
    size = "small",
    srcs = ["framework/contrib_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
    ],
    main = "framework/contrib_test.py",
    tags = [
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "build_info_test",
    size = "small",
    srcs = [
        "platform/build_info.py",
        "platform/build_info_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":platform",
    ],
    main = "platform/build_info_test.py",
    tags = [
        "no_pip",
        "notap",
    ],
)

tf_py_test(
    name = "benchmark_test",
    size = "small",
    srcs = [
        "platform/benchmark.py",
        "platform/benchmark_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":platform",
    ],
    main = "platform/benchmark_test.py",
    tags = [
        "no_pip",
    ],
)

tf_py_test(
    name = "proto_test",
    size = "small",
    srcs = ["framework/proto_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
    main = "framework/proto_test.py",
)

tf_gen_op_wrapper_private_py(
    name = "functional_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["ops/functional_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        ":functional_ops_gen",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":util",
        ":variable_scope",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "function_test",
    size = "medium",
    srcs = ["framework/function_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":clip_ops",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradients",
        ":init_ops",
        ":logging_ops",
        ":logging_ops_gen",
        ":math_ops",
        ":nn_ops",
        ":platform",
        ":random_ops",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    shard_count = 10,
    tags = [
        "noasan",
        "optonly",
    ],
    xla_enable_strict_auto_jit = True,
)

tf_py_test(
    name = "framework_versions_test",
    size = "small",
    srcs = ["framework/versions_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
    ],
    main = "framework/versions_test.py",
)

tf_py_test(
    name = "framework_importer_test",
    size = "large",
    srcs = ["framework/importer_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":random_ops",
        ":test_ops",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/importer_test.py",
)

filegroup(
    name = "meta_graph_testdata",
    srcs = [
        "framework/testdata/metrics_export_meta_graph.pb",
    ],
    visibility = ["//visibility:public"],
)

tf_py_test(
    name = "framework_meta_graph_test",
    size = "small",
    srcs = ["framework/meta_graph_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":metrics",
        ":nn_ops",
        ":platform",
        ":random_ops",
        ":training",
        ":variables",
    ],
    data = ["//tensorflow/python:meta_graph_testdata"],
    main = "framework/meta_graph_test.py",
    tags = [
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "framework_traceable_stack_test",
    size = "small",
    srcs = ["framework/traceable_stack_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":platform_test",
        ":test_ops",
        ":traceable_stack",
        ":util",
    ],
    main = "framework/traceable_stack_test.py",
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "framework/test_ops.py",
    deps = [":test_ops_kernels"],
)

cc_library(
    name = "test_ops_kernels",
    srcs = ["framework/test_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "test_ops_2",
    out = "framework/test_ops_2.py",
    deps = [":test_ops_2_kernels"],
)

cc_library(
    name = "test_ops_2_kernels",
    srcs = ["framework/test_ops_2.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_py_test(
    name = "framework_common_shapes_test",
    size = "small",
    srcs = ["framework/common_shapes_test.py"],
    additional_deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/common_shapes_test.py",
)

tf_py_test(
    name = "framework_ops_test",
    size = "small",
    srcs = ["framework/ops_test.py"],
    additional_deps = [
        ":cond_v2",
        ":control_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":resources",
        ":test_ops",
        ":test_ops_2",
        ":util",
        ":variable_scope",
        ":variables",
        ":while_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
    ],
    main = "framework/ops_test.py",
    tags = ["no_pip"],  # test_ops_2 is not available in pip.
)

tf_py_test(
    name = "framework_ops_enable_eager_test",
    size = "small",
    srcs = ["framework/ops_enable_eager_test.py"],
    additional_deps = [
        ":framework",
        ":platform_test",
        "//tensorflow/python/eager:context",
    ],
    main = "framework/ops_enable_eager_test.py",
)

tf_py_test(
    name = "framework_tensor_shape_test",
    size = "small",
    srcs = ["framework/tensor_shape_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
        "@absl_py//absl/testing:parameterized",
    ],
    main = "framework/tensor_shape_test.py",
)

tf_py_test(
    name = "framework_type_spec_test",
    size = "small",
    srcs = ["framework/type_spec_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":type_spec",
        "//third_party/py/numpy",
    ],
    main = "framework/type_spec_test.py",
)

tf_py_test(
    name = "framework_tensor_spec_test",
    size = "small",
    srcs = ["framework/tensor_spec_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":tensor_spec",
        "//third_party/py/numpy",
    ],
    main = "framework/tensor_spec_test.py",
)

tf_py_test(
    name = "framework_sparse_tensor_test",
    size = "small",
    srcs = ["framework/sparse_tensor_test.py"],
    additional_deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/sparse_tensor_test.py",
)

tf_py_test(
    name = "framework_device_spec_test",
    size = "small",
    srcs = ["framework/device_spec_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/device_spec_test.py",
)

tf_py_test(
    name = "framework_device_test",
    size = "small",
    srcs = ["framework/device_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/device_test.py",
)

tf_py_test(
    name = "framework_random_seed_test",
    size = "small",
    srcs = ["framework/random_seed_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework",
    ],
    main = "framework/random_seed_test.py",
)

tf_py_test(
    name = "framework_tensor_shape_div_test",
    size = "small",
    srcs = ["framework/tensor_shape_div_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/tensor_shape_div_test.py",
)

tf_py_test(
    name = "framework_tensor_util_test",
    size = "small",
    srcs = ["framework/tensor_util_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":state_ops_gen",
        "//third_party/py/numpy",
    ],
    main = "framework/tensor_util_test.py",
    tags = ["no_windows"],
)

tf_py_test(
    name = "framework_test_util_test",
    size = "small",
    srcs = ["framework/test_util_test.py"],
    additional_deps = [
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":random_ops",
        ":resource_variable_ops",
        ":session",
        ":test_ops",
        ":variables",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
    ],
    main = "framework/test_util_test.py",
    tags = ["no_windows"],
)

tf_py_test(
    name = "framework_dtypes_test",
    size = "small",
    srcs = ["framework/dtypes_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
    ],
    main = "framework/dtypes_test.py",
)

tf_py_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["framework/op_def_library_test.py"],
    additional_deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":test_ops",
    ],
)

tf_py_test(
    name = "framework_kernels_test",
    size = "small",
    srcs = ["framework/kernels_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":kernels",
        ":platform_test",
        ":test_ops",
    ],
    main = "framework/kernels_test.py",
)

tf_gen_op_wrapper_private_py(
    name = "array_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
    deps = [
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/core:array_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "bitwise_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "boosted_trees_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:boosted_trees_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "tensor_forest_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:tensor_forest_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "summary_ops_gen",
    visibility = ["//tensorflow:__subpackages__"],
    deps = ["//tensorflow/core:summary_ops_op_lib"],
)

tf_gen_op_wrapper_private_py(
    name = "audio_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/contrib/framework:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "cudnn_rnn_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "candidate_sampling_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "checkpoint_ops_gen",
    visibility = [
        "//tensorflow/contrib/framework:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "clustering_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:clustering_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "collective_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:collective_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "control_flow_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
    deps = [
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "ctc_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "data_flow_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "dataset_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "experimental_dataset_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "image_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "io_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/training/tracking:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "linalg_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "logging_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "lookup_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/contrib/lookup:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "batch_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "batch_ops",
    srcs = [
        "ops/batch_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":batch_ops_gen",
    ],
)

py_test(
    name = "batch_ops_test",
    size = "small",
    srcs = ["ops/batch_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_pip",
        "nomac",
    ],
    deps = [
        ":batch_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradients",
        "//tensorflow/python:script_ops",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "manip_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "math_ops_gen",
    visibility = [
        "//learning/brain/google/python/ops:__pkg__",
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "nn_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/tools:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "parsing_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "random_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "stateful_random_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "resource_variable_ops_gen",
    visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "stateless_random_ops_gen",
    visibility = [
        "//tensorflow/python/data/experimental/ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "list_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "script_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "sdca_ops_gen",
    visibility = [
        "//tensorflow/contrib/linear_optimizer:__pkg__",
        "//tensorflow_estimator/python/estimator/canned/linear_optimizer:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "set_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "state_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/contrib/framework:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "sparse_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "spectral_ops_gen",
    visibility = ["//tensorflow/python/ops/signal:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "string_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
    require_shape_functions = False,
)

tf_gen_op_wrapper_private_py(
    name = "training_ops_gen",
    out = "training/gen_training_ops.py",
)

tf_gen_op_wrapper_private_py(
    name = "ragged_array_ops_gen",
    visibility = [
        "//learning/brain/contrib/text:__pkg__",
        "//learning/brain/contrib/text/python/ragged:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "ragged_math_ops_gen",
    visibility = [
        "//learning/brain/contrib/text:__pkg__",
        "//learning/brain/contrib/text/python/ragged:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "ragged_conversion_ops_gen",
    visibility = [
        "//learning/brain/contrib/text:__pkg__",
        "//learning/brain/contrib/text/python/ragged:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "rnn_ops_gen",
    visibility = [
        "//tensorflow/contrib/rnn:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "tpu_ops_gen",
    visibility = [
        "//smartass/brain/configure/python:__pkg__",
        "//tensorflow/contrib/tpu:__pkg__",
        "//tensorflow/python/tpu:__pkg__",
    ],
    deps = [
        "//tensorflow/core:tpu_configuration_ops_op_lib",
        "//tensorflow/core:tpu_cross_replica_ops_op_lib",
        "//tensorflow/core:tpu_embedding_ops_op_lib",
        "//tensorflow/core:tpu_functional_ops_op_lib",
        "//tensorflow/core:tpu_heartbeat_ops_op_lib",
        "//tensorflow/core:tpu_host_compute_ops_op_lib",
        "//tensorflow/core:tpu_infeed_ops_op_lib",
        "//tensorflow/core:tpu_ordinal_selector_ops_op_lib",
        "//tensorflow/core:tpu_outfeed_ops_op_lib",
        "//tensorflow/core:tpu_replication_ops_op_lib",
    ],
)

py_library(
    name = "array_grad",
    srcs = ["ops/array_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
    ],
)

py_library(
    name = "array_ops",
    srcs = [
        "ops/array_ops.py",
        "ops/inplace_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops_gen",
        ":common_shapes",
        ":constant_op",
        ":dtypes",
        ":framework_ops",
        ":math_ops_gen",
        ":sparse_tensor",
        ":tensor_shape",
        ":tensor_util",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "bitwise_ops",
    srcs = ["ops/bitwise_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bitwise_ops_gen",
        ":framework",
        ":util",
    ],
)

py_library(
    name = "boosted_trees_ops",
    srcs = ["ops/boosted_trees_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_ops_gen",
        ":framework",
        ":ops",
        ":training",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
    ],
)

py_library(
    name = "tensor_forest_ops",
    srcs = ["ops/tensor_forest_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":ops",
        ":tensor_forest_ops_gen",
        ":training",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
    ],
)

py_library(
    name = "optional_grad",
    srcs = ["ops/optional_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_ops",
    ],
)

py_library(
    name = "sets",
    srcs = [
        "ops/sets.py",
        "ops/sets_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":set_ops_gen",
        ":util",
    ],
)

py_library(
    name = "candidate_sampling_ops",
    srcs = ["ops/candidate_sampling_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops_gen",
        ":framework",
        ":math_ops",
    ],
)

py_library(
    name = "check_ops",
    srcs = ["ops/check_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_tensor",
        ":tensor_util",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clip_ops",
    srcs = ["ops/clip_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_ops_gen",
        ":numerics",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["ops/clip_ops_test.py"],
    additional_deps = [
        ":client_testlib",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clustering_ops",
    srcs = ["ops/clustering_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clustering_ops_gen",
        ":framework",
        ":ops",
        ":training",
    ],
)

tf_py_test(
    name = "clustering_ops_test",
    size = "medium",
    srcs = ["ops/clustering_ops_test.py"],
    additional_deps = [
        ":client_testlib",
        ":clustering_ops",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "collective_ops",
    srcs = ["ops/collective_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":collective_ops_gen",
        ":framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "collective_ops_test",
    size = "small",
    srcs = ["ops/collective_ops_test.py"],
    additional_deps = [
        ":client_testlib",
        ":collective_ops",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "control_flow_grad",
    srcs =
        ["ops/control_flow_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":control_flow_ops_gen",
        ":control_flow_util",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

# Note: targets depending on this should also depend on ":cond_v2" and ":while_v2".
# See b/118513001.
py_library(
    name = "control_flow_ops",
    srcs = ["ops/control_flow_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":constant_op",
        ":control_flow_ops_gen",
        ":control_flow_util",
        ":dtypes",
        ":framework_ops",
        ":logging_ops_gen",
        ":math_ops",
        ":platform",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":tf2",
        ":tf_should_use",
        ":util",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "control_flow_util",
    srcs = ["ops/control_flow_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":platform",
        ":util",
    ],
)

py_library(
    name = "control_flow_util_v2",
    srcs = ["ops/control_flow_util_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_util",
        ":framework_ops",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
    ],
)

py_library(
    name = "cond_v2",
    srcs = [
        "ops/cond_v2.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":c_api_util",
        ":control_flow_util_v2",
        ":framework_ops",
        ":function",
        ":function_def_to_graph",
        ":functional_ops_gen",
        ":gradients",
        ":gradients_util",
        ":graph_to_function_def",
        ":pywrap_tensorflow",
        ":util",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:function",
    ],
)

py_library(
    name = "while_v2",
    srcs = [
        "ops/while_v2.py",
        "ops/while_v2_indexed_slices_rewriter.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":dtypes",
        ":framework_ops",
        ":function_def_to_graph",
        ":functional_ops_gen",
        ":gradients_util",
        ":list_ops",
        ":tensor_array_ops",
        ":tensor_shape",
        ":tensor_util",
        ":util",
        "//tensorflow/python/eager:function",
    ],
)

py_library(
    name = "ctc_ops",
    srcs = ["ops/ctc_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":ctc_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":nn_grad",
    ],
)

py_library(
    name = "cudnn_rnn_grad",
    srcs = ["ops/cudnn_rnn_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        "//tensorflow/python:cudnn_rnn_ops_gen",
    ],
)

py_library(
    name = "data_flow_grad",
    srcs = ["ops/data_flow_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "data_flow_ops",
    srcs = ["ops/data_flow_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":data_flow_ops_gen",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":random_seed",
        ":tensor_util",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_library(
    name = "embedding_ops",
    srcs = ["ops/embedding_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        ":resource_variable_ops",
        ":sparse_ops",
        ":tensor_shape",
        ":variables",
    ],
)

py_library(
    name = "gradients",
    srcs = [
        "ops/custom_gradient.py",
        "ops/gradients.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gradients_impl",
        ":gradients_util",
        ":unconnected_gradients",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "gradients_impl",
    srcs = [
        "ops/gradients_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":bitwise_ops",
        ":check_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":control_flow_util",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":image_grad",
        ":linalg_grad",
        ":linalg_ops",
        ":logging_ops",
        ":manip_grad",
        ":manip_ops",
        ":math_grad",
        ":math_ops",
        ":optional_grad",
        ":platform",
        ":random_grad",
        ":tensor_array_ops",
        ":unconnected_gradients",
        ":util",
    ],
)

py_library(
    name = "gradients_util",
    srcs = [
        "ops/gradients_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":control_flow_state",
        ":control_flow_util",
        ":default_gradient",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":functional_ops",
        ":math_ops",
        ":platform",
        ":resource_variable_ops",
        ":tensor_util",
        ":unconnected_gradients",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "default_gradient",
    srcs = [
        "ops/default_gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":resource_variable_ops",
    ],
)

py_library(
    name = "control_flow_state",
    srcs = [
        "ops/control_flow_state.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":control_flow_util",
        ":data_flow_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":tensor_util",
    ],
)

py_library(
    name = "unconnected_gradients",
    srcs = ["ops/unconnected_gradients.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
    ],
)

py_library(
    name = "histogram_ops",
    srcs = ["ops/histogram_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "image_grad",
    srcs = ["ops/image_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":image_ops_gen",
    ],
)

py_library(
    name = "image_ops",
    srcs = [
        "ops/image_ops.py",
        "ops/image_ops_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":clip_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":image_ops_gen",
        ":math_ops",
        ":nn",
        ":nn_ops_gen",
        ":random_ops",
        ":string_ops",
        ":util",
        ":variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "init_ops",
    srcs = ["ops/init_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":math_ops",
        ":random_ops",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "init_ops_v2",
    srcs = ["ops/init_ops_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":math_ops",
        ":random_ops",
        ":stateless_random_ops",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "initializers_ns",
    srcs = ["ops/initializers_ns.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":init_ops",
        ":variables",
    ],
)

py_library(
    name = "io_ops",
    srcs = ["ops/io_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":io_ops_gen",
        ":lib",
    ],
)

py_library(
    name = "linalg_grad",
    srcs = ["ops/linalg_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":linalg_ops",
        ":math_ops",
        "//tensorflow/python/ops/linalg:linalg_impl",
    ],
)

py_library(
    name = "linalg_ops",
    srcs = ["ops/linalg_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":map_fn",
        ":math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "linalg_ops_impl",
    srcs = ["ops/linalg_ops_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "manip_grad",
    srcs = ["ops/manip_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":manip_ops",
    ],
)

py_library(
    name = "manip_ops",
    srcs = ["ops/manip_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":manip_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "logging_ops",
    srcs = ["ops/logging_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":logging_ops_gen",
        ":platform",
        ":string_ops",
        ":util",
        "//tensorflow/python/compat",
    ],
)

py_library(
    name = "lookup_ops",
    srcs = ["ops/lookup_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":lookup_ops_gen",
        ":math_ops",
        ":sparse_tensor",
        ":string_ops",
        ":util",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_library(
    name = "loss_scale",
    srcs = ["training/experimental/loss_scale.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["training/experimental/loss_scale_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss_scale",
        "//tensorflow/python/distribute:distribute_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "loss_scale_optimizer_test",
    size = "small",
    srcs = ["training/experimental/loss_scale_optimizer_test.py"],
    python_version = "PY2",
    deps = [
        ":loss_scale_optimizer",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/keras/mixed_precision/experimental:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "loss_scale_test",
    size = "medium",
    srcs = ["training/experimental/loss_scale_test.py"],
    python_version = "PY2",
    deps = [
        ":loss_scale",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "mixed_precision_global_state",
    srcs = ["training/experimental/mixed_precision_global_state.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "mixed_precision",
    srcs = ["training/experimental/mixed_precision.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":config",
        ":loss_scale",
        ":loss_scale_optimizer",
        ":mixed_precision_global_state",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "mixed_precision_test",
    size = "small",
    srcs = ["training/experimental/mixed_precision_test.py"],
    additional_deps = [
        ":mixed_precision",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_rocm",
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "math_grad",
    srcs = ["ops/math_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":math_ops_gen",
        ":tensor_util",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "op_selector",
    srcs = ["ops/op_selector.py"],
    srcs_version = "PY2AND3",
    deps = [":framework_ops"],
)

py_library(
    name = "math_ops",
    srcs = ["ops/math_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "constant_op",
        ":array_ops",
        ":common_shapes",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":graph_util",
        ":math_ops_gen",
        ":nn_ops_gen",
        ":sparse_ops_gen",
        ":sparse_tensor",
        ":state_ops",
        ":state_ops_gen",
        ":tensor_shape",
        ":util",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "resources",
    srcs = ["ops/resources.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":tf_should_use",
    ],
)

py_library(
    name = "resource_variable_ops",
    srcs = ["ops/resource_variable_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":tensor_shape",
        ":util",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "critical_section_ops",
    srcs = ["ops/critical_section_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":tensor_array_ops",
        ":util",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "list_ops",
    srcs = ["ops/list_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":list_ops_gen",
    ],
)

py_library(
    name = "nn",
    srcs = [
        "ops/nn.py",
        "ops/nn_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops",
        ":ctc_ops",
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":nn_ops_gen",
        ":rnn",
        ":sparse_ops",
        ":util",
        ":variables",
    ],
)

py_library(
    name = "nn_grad",
    srcs = ["ops/nn_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops",
        ":tensor_util",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "nn_ops",
    srcs = ["ops/nn_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":graph_util",
        ":math_ops",
        ":nn_ops_gen",
        ":platform",
        ":random_ops",
        ":tensor_shape",
        ":tensor_util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "numerics",
    srcs = ["ops/numerics.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["ops/parsing_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":parsing_ops_gen",
        ":sparse_ops",
    ],
)

py_library(
    name = "partitioned_variables",
    srcs = ["ops/partitioned_variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":platform",
        ":variable_scope",
    ],
)

py_library(
    name = "random_grad",
    srcs = ["ops/random_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":random_ops_gen",
    ],
)

py_library(
    name = "random_ops",
    srcs = ["ops/random_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":random_ops_gen",
        ":random_seed",
    ],
)

py_library(
    name = "stateful_random_ops",
    srcs = ["ops/stateful_random_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":stateful_random_ops_gen",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["ops/stateful_random_ops_test.py"],
    additional_deps = [
        ":stateful_random_ops",
        ":client_testlib",
        ":logging_ops",
        ":random_ops_gen",
        "//tensorflow/python/kernel_tests/random:util",
        "//tensorflow/python/distribute:mirrored_strategy",
    ],
    xla_enable_strict_auto_jit = False,
)

py_library(
    name = "stateless_random_ops",
    srcs = ["ops/stateless_random_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":random_ops",
        ":stateless_random_ops_gen",
    ],
)

py_library(
    name = "rnn",
    srcs = ["ops/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":control_flow_util",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":rnn_cell",
        ":tensor_array_ops",
        ":util",
        ":variable_scope",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = [
        "ops/rnn_cell.py",
        "ops/rnn_cell_impl.py",
        "ops/rnn_cell_wrapper_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        ":init_ops",
        ":math_ops",
        ":nn_ops",
        ":partitioned_variables",
        ":random_ops",
        ":util",
        ":variable_scope",
        ":variables",
    ],
)

py_library(
    name = "script_ops",
    srcs = ["ops/script_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":script_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sdca_ops",
    srcs = ["ops/sdca_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":sdca_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "session_ops",
    srcs = ["ops/session_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        ":framework_for_generated_wrappers",
        ":util",
    ],
)

py_library(
    name = "sparse_grad",
    srcs = ["ops/sparse_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
        ":sparse_ops_gen",
    ],
)

py_library(
    name = "sparse_ops",
    srcs = ["ops/sparse_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops_gen",
        ":util",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_ops_test",
    srcs = ["ops/sparse_ops_test.py"],
    additional_deps = [
        ":array_grad",
        ":constant_op",
        ":dtypes",
        ":framework_test_lib",
        ":sparse_ops",
        ":sparse_tensor",
        ":sparse_grad",
        ":gradient_checker_v2",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "sort_ops",
    srcs = ["ops/sort_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":math_ops",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sort_ops_test",
    srcs = ["ops/sort_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":random_ops",
        ":sort_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "spectral_ops_test_util",
    srcs = ["ops/spectral_ops_test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework_ops",
    ],
)

py_library(
    name = "confusion_matrix",
    srcs = ["ops/confusion_matrix.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
    ],
)

py_library(
    name = "weights_broadcast_ops",
    srcs = [
        "ops/weights_broadcast_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":math_ops",
        ":sets",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "ops/metrics.py",
        "ops/metrics_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":confusion_matrix",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":sets",
        ":sparse_ops",
        ":state_ops",
        ":util",
        ":variable_scope",
        ":variables",
        ":weights_broadcast_ops",
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

py_library(
    name = "special_math_ops",
    srcs = ["ops/special_math_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "@opt_einsum_archive//:opt_einsum",
    ],
)

py_library(
    name = "rnn_grad",
    srcs = ["ops/rnn_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        "//tensorflow/python:rnn_ops_gen",
    ],
)

py_library(
    name = "standard_ops",
    srcs = ["ops/standard_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":batch_ops",
        ":check_ops",
        ":clip_ops",
        ":confusion_matrix",
        ":control_flow_ops",
        ":critical_section_ops",
        ":cudnn_rnn_grad",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradients",
        ":histogram_ops",
        ":init_ops",
        ":io_ops",
        ":linalg_ops",
        ":logging_ops",
        ":lookup_ops",
        ":manip_grad",
        ":manip_ops",
        ":math_grad",
        ":math_ops",
        ":numerics",
        ":parsing_ops",
        ":partitioned_variables",
        ":proto_ops",
        ":random_ops",
        ":rnn_grad",
        ":script_ops",
        ":session_ops",
        ":sort_ops",
        ":sparse_grad",
        ":sparse_ops",
        ":special_math_ops",
        ":state_grad",
        ":state_ops",
        ":stateful_random_ops",
        ":stateless_random_ops",
        ":string_ops",
        ":template",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":util",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/ragged",
    ],
)

py_library(
    name = "state_grad",
    srcs = ["ops/state_grad.py"],
    srcs_version = "PY2AND3",
    deps = [":framework_for_generated_wrappers"],
)

py_library(
    name = "state_ops",
    srcs = ["ops/state_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_ops",
        ":math_ops_gen",
        ":resource_variable_ops_gen",
        ":state_ops_gen",
        ":tensor_shape",
        ":util",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["ops/string_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":string_ops_gen",
        ":util",
    ],
)

py_library(
    name = "summary_ops_v2",
    srcs = ["ops/summary_ops_v2.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":resource_variable_ops",
        ":smart_cond",
        ":summary_op_util",
        ":summary_ops_gen",
        ":tensor_util",
        ":training_util",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:profiler",
        "@six_archive//:six",
    ],
)

py_library(
    name = "template",
    srcs = ["ops/template.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":platform",
        ":util",
        ":variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
    ],
)

py_library(
    name = "tensor_array_grad",
    srcs = ["ops/tensor_array_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":tensor_array_ops",
    ],
)

py_library(
    name = "tensor_array_ops",
    srcs = ["ops/tensor_array_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":dtypes",
        ":errors",
        ":framework_ops",
        ":list_ops",
        ":math_ops",
        ":tensor_shape",
        ":tensor_util",
        ":tf2",
        ":tf_should_use",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "variable_scope",
    srcs = ["ops/variable_scope.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":init_ops",
        ":platform",
        ":resource_variable_ops",
        ":tensor_shape",
        ":tf2",
        ":util",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "@six_archive//:six",
    ],
)

py_library(
    name = "variables",
    srcs = ["ops/variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":state_ops",
        ":tensor_shape",
        ":tf_should_use",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/tracking:base",
    ],
)

py_library(
    name = "gradient_checker",
    srcs = ["ops/gradient_checker.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "gradient_checker_v2",
    srcs = ["ops/gradient_checker_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        "//third_party/py/numpy",
    ],
)

# This target is deprecated.
py_library(
    name = "ops",
    srcs = ["user_ops/user_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":user_ops_gen",
        ":util",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "bitwise_ops_test",
    size = "small",
    srcs = ["ops/bitwise_ops_test.py"],
    additional_deps = [
        ":bitwise_ops",
        ":constant_op",
        ":dtypes",
        ":framework_test_lib",
    ],
    tags = ["no_windows"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "control_flow_ops_test",
    size = "small",
    srcs = ["ops/control_flow_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":cond_v2",
        ":control_flow_ops",
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":init_ops",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        ":while_v2",
        "//tensorflow/python/eager:def_function",
    ],
    shard_count = 2,
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "gradient_checker_test",
    size = "medium",
    srcs = ["ops/gradient_checker_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

py_test(
    name = "op_selector_test",
    srcs = ["ops/op_selector_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":constant_op",
        ":framework_ops",
        ":math_ops",
        ":op_selector",
    ],
)

cuda_py_test(
    name = "gradient_checker_v2_test",
    size = "medium",
    srcs = ["ops/gradient_checker_v2_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["ops/gradients_test.py"],
    additional_deps = [
        ":array_grad",
        ":array_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":functional_ops",
        ":gradients",
        ":list_ops",
        ":math_grad",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform_test",
        ":state_grad",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":test_ops",
        ":unconnected_gradients",
        ":variable_scope",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python/keras:engine",
    ],
    tags = ["no_oss"],  # b/118709825
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "histogram_ops_test",
    size = "small",
    srcs = ["ops/histogram_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":histogram_ops",
        ":init_ops",
        ":variables",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "image_grad_test",
    size = "medium",
    srcs = ["ops/image_grad_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":image_ops",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["ops/image_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":image_ops",
        ":io_ops",
        ":math_ops",
        ":platform_test",
        ":random_ops",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    data = ["//tensorflow/core:image_testdata"],
    shard_count = 5,
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "init_ops_test",
    size = "small",
    srcs = ["ops/init_ops_test.py"],
    additional_deps = [
        ":client_testlib",
        ":init_ops",
        ":framework_ops",
        ":resource_variable_ops",
        "//third_party/py/numpy",
        "//tensorflow/python/eager:context",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "init_ops_v2_test",
    size = "medium",
    srcs = ["ops/init_ops_v2_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":init_ops_v2",
        ":random_ops",
        ":framework_ops",
        "//third_party/py/numpy",
        "//tensorflow/python/eager:context",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "math_grad_test",
    size = "small",
    srcs = ["ops/math_grad_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execution_callbacks",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows_gpu"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "math_ops_test",
    size = "small",
    srcs = ["ops/math_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":variables",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows_gpu"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "nn_batchnorm_test",
    size = "medium",
    srcs = ["ops/nn_batchnorm_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        "//third_party/py/numpy",
    ],
    shard_count = 4,
    tags = ["no_windows"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "nn_fused_batchnorm_test",
    size = "medium",
    srcs = ["ops/nn_fused_batchnorm_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//third_party/py/numpy",
    ],
    shard_count = 16,
    tags = ["no_rocm"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "nn_test",
    size = "medium",
    srcs = ["ops/nn_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn",
        ":nn_grad",
        ":nn_ops",
        ":partitioned_variables",
        ":variable_scope",
        ":variables",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows"],
    xla_enable_strict_auto_jit = True,
)

py_test(
    name = "nn_loss_scaling_utilities_test",
    size = "small",
    srcs = ["ops/nn_loss_scaling_utilities_test.py"],
    python_version = "PY2",
    deps = [
        ":client_testlib",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "nn_xent_test",
    size = "medium",
    srcs = ["ops/nn_xent_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "special_math_ops_test",
    size = "small",
    srcs = ["ops/special_math_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":special_math_ops",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows_gpu"],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "training_lib",
    srcs = glob(
        ["training/**/*.py"],
        exclude = [
            "**/*test*",
            "training/tracking/**/*.py",
            "training/saving/**/*.py",
            # The following targets have their own build rules (same name as the
            # file):
            "training/basic_session_run_hooks.py",
            "training/checkpoint_management.py",
            "training/distribute.py",
            "training/distribution_strategy_context.py",
            "training/saver.py",
            "training/session_run_hook.py",
            "training/training_util.py",
        ],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":basic_session_run_hooks",
        ":checkpoint_management",
        ":checkpoint_ops_gen",
        ":client",
        ":control_flow_ops",
        ":data_flow_ops",
        ":device",
        ":device_spec",
        ":distribute",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":gradients",
        ":init_ops",
        ":io_ops",
        ":layers_util",
        ":lookup_ops",
        ":loss_scale",
        ":loss_scale_optimizer",
        ":math_ops",
        ":mixed_precision",
        ":platform",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":resources",
        ":saver",
        ":sdca_ops",
        ":session",
        ":session_run_hook",
        ":sparse_ops",
        ":sparse_tensor",
        ":state_ops",
        ":summary",
        ":training_ops_gen",
        ":training_util",
        ":util",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras/optimizer_v2:learning_rate_schedule",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "training",
    srcs_version = "PY2AND3",
    deps = [
        ":training_lib",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:python_state",
        "//tensorflow/python/training/tracking:util",
    ],
)

# Dependency added and used by ClusterResolvers to avoid circular dependency between keras, distribute, and training.
py_library(
    name = "training_server_lib",
    srcs = ["training/server_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":pywrap_tensorflow",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "checkpoint_management",
    srcs = ["training/checkpoint_management.py"],
    deps = [
        ":errors",
        ":lib",
        ":platform",
        ":protos_all_py",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "session_run_hook",
    srcs = ["training/session_run_hook.py"],
    srcs_version = "PY2AND3",
    deps = [":util"],
)

py_library(
    name = "basic_session_run_hooks",
    srcs = ["training/basic_session_run_hooks.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":framework",
        ":platform",
        ":protos_all_py",
        ":session_run_hook",
        ":training_util",
        ":util",
    ],
)

py_library(
    name = "saver",
    srcs = ["training/saver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":checkpoint_management",
        ":constant_op",
        ":control_flow_ops",
        ":device",
        ":errors",
        ":framework",
        ":framework_ops",
        ":io_ops",
        ":io_ops_gen",
        ":platform",
        ":pywrap_tensorflow",
        ":resource_variable_ops",
        ":session",
        ":state_ops",
        ":string_ops",
        ":training_util",
        ":util",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/training/tracking:base",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "distribute",
    srcs = [
        "training/distribute.py",
        "training/distribution_strategy_context.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

tf_py_test(
    name = "evaluation_test",
    size = "small",
    srcs = ["training/evaluation_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":metrics",
        ":platform",
        ":state_ops",
        ":summary",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops/losses",
    ],
    shard_count = 3,
    tags = [
        "manual",
        "notap",  # Disabling until b/33000128 and b/33040312 are fixed.
    ],
)

py_library(
    name = "client",
    srcs = [
        "client/client_lib.py",
        "client/device_lib.py",
        "client/timeline.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
        ":session",
        ":session_ops",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "util",
    srcs = glob(
        ["util/**/*.py"],
        exclude = [
            "util/example_parser*",
            "util/tf_should_use.py",
            "util/**/*_test.py",
        ],
    ),
    srcs_version = "PY2AND3",
    visibility = visibility + [
        "//tensorflow:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
    ],
    deps = [
        "//tensorflow/tools/compatibility:all_renames_v2",
        "//third_party/py/numpy",
        "@com_google_protobuf//:protobuf_python",
        "@org_python_pypi_backports_weakref",
        "@six_archive//:six",
    ],
)

# Placeholder for intenal nest_test comments.
tf_py_test(
    name = "util_nest_test",
    size = "small",
    srcs = ["util/nest_test.py"],
    additional_deps = [":util_nest_test_main_lib"],
    main = "util/nest_test.py",
)

py_library(
    name = "util_nest_test_main_lib",
    testonly = True,
    srcs = ["util/nest_test.py"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "util_serialization_test",
    size = "small",
    srcs = ["util/serialization_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
    ],
    main = "util/serialization_test.py",
)

tf_py_test(
    name = "function_utils_test",
    srcs = ["util/function_utils_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
    ],
)

tf_py_test(
    name = "tf_contextlib_test",
    size = "small",
    srcs = ["util/tf_contextlib_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
    ],
)

tf_py_test(
    name = "tf_decorator_test",
    size = "small",
    srcs = ["util/tf_decorator_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
    ],
)

py_library(
    name = "tf_should_use",
    srcs = ["util/tf_should_use.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "tf_should_use_test",
    size = "small",
    srcs = ["util/tf_should_use_test.py"],
    additional_deps = [
        ":client_testlib",
        ":tf_should_use",
    ],
)

tf_py_test(
    name = "tf_inspect_test",
    size = "small",
    srcs = ["util/tf_inspect_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
    ],
)

py_library(
    name = "util_example_parser_configuration",
    srcs = ["util/example_parser_configuration.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "lock_util_test",
    size = "small",
    srcs = ["util/lock_util_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
        "@absl_py//absl/testing:parameterized",
    ],
    main = "util/lock_util_test.py",
)

tf_py_test(
    name = "deprecation_wrapper_test",
    size = "small",
    srcs = ["util/deprecation_wrapper_test.py"],
    additional_deps = [
        ":client_testlib",
        ":util",
        "@six_archive//:six",
        "//tensorflow/tools/compatibility:all_renames_v2",
    ],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(
        ["**/*.proto"],
        exclude = [
            "util/protobuf/compare_test.proto",
            "framework/cpp_shape_inference.proto",
        ],
    ),
)

tf_proto_library_py(
    name = "compare_test_proto",
    testonly = 1,
    srcs = ["util/protobuf/compare_test.proto"],
)

tf_proto_library(
    name = "cpp_shape_inference_proto",
    srcs = ["framework/cpp_shape_inference.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    visibility = ["//tensorflow:internal"],
)

tf_py_test(
    name = "protobuf_compare_test",
    size = "small",
    srcs = ["util/protobuf/compare_test.py"],
    additional_deps = [
        ":compare_test_proto_py",
        ":platform_test",
        ":util",
        "@six_archive//:six",
    ],
    main = "util/protobuf/compare_test.py",
    tags = ["no_pip"],  # compare_test_pb2 proto is not available in pip.
)

tf_py_test(
    name = "util_example_parser_configuration_test",
    size = "small",
    srcs = ["util/example_parser_configuration_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":parsing_ops",
        ":util_example_parser_configuration",
    ],
    main = "util/example_parser_configuration_test.py",
)

tf_py_test(
    name = "events_writer_test",
    size = "small",
    srcs = ["client/events_writer_test.py"],
    additional_deps = [
        ":errors",
        ":framework_test_lib",
        ":lib",
        ":platform_test",
        ":util",
    ],
)

py_library(
    name = "device_lib",
    srcs = ["client/device_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
    ],
)

cc_library(
    name = "cpp_shape_inference",
    srcs = ["framework/cpp_shape_inference.cc"],
    hdrs = ["framework/cpp_shape_inference.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    visibility = ["//visibility:public"],
    deps = [
        ":cpp_shape_inference_proto_cc",
        ":numpy_lib",
        ":py_func_lib",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python:ndarray_tensor",
    ],
)

cuda_py_tests(
    name = "device_lib_test",
    size = "small",
    srcs = [
        "client/device_lib_test.py",
    ],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
    xla_enable_strict_auto_jit = True,
)

cc_library(
    name = "session_ref",
    srcs = ["client/session_ref.cc"],
    hdrs = ["client/session_ref.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:replay_log_proto_cc",
    ],
)

tf_cuda_library(
    name = "tf_session_helper",
    srcs = ["client/tf_session_helper.cc"],
    hdrs = ["client/tf_session_helper.h"],
    deps = [
        ":construction_fails_op",
        ":ndarray_tensor",
        ":ndarray_tensor_bridge",
        ":numpy_lib",
        ":safe_ptr",
        ":session_ref",
        ":test_ops_kernels",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

py_library(
    name = "pywrap_tensorflow",
    srcs = [
        "pywrap_tensorflow.py",
    ] + if_static(
        ["pywrap_dlopen_global_flags.py"],
        # Import will fail, indicating no global dlopen flags
        otherwise = [],
    ),
    srcs_version = "PY2AND3",
    deps = [":pywrap_tensorflow_internal"],
)

tf_py_wrap_cc(
    name = "pywrap_tensorflow_internal",
    srcs = ["tensorflow.i"],
    swig_includes = [
        "client/device_lib.i",
        "client/events_writer.i",
        "client/tf_session.i",
        "client/tf_sessionrun_wrapper.i",
        "framework/cpp_shape_inference.i",
        "framework/python_op_gen.i",
        "grappler/cluster.i",
        "grappler/cost_analyzer.i",
        "grappler/graph_analyzer.i",
        "grappler/item.i",
        "grappler/model_analyzer.i",
        "grappler/tf_optimizer.i",
        "lib/core/bfloat16.i",
        "lib/core/py_exception_registry.i",
        "lib/core/py_func.i",
        "lib/core/strings.i",
        "lib/io/file_io.i",
        "lib/io/py_record_reader.i",
        "lib/io/py_record_writer.i",
        "platform/base.i",
        "platform/stacktrace_handler.i",
        "pywrap_tfe.i",
        "training/quantize_training.i",
        "util/kernel_registry.i",
        "util/port.i",
        "util/py_checkpoint_reader.i",
        "util/stat_summarizer.i",
        "util/tfprof.i",
        "util/transform_graph.i",
        "util/util.i",
    ],
    # add win_def_file for pywrap_tensorflow
    win_def_file = select({
        "//tensorflow:windows": ":pywrap_tensorflow_filtered_def_file",
        "//conditions:default": None,
    }),
    deps = [
        ":bfloat16_lib",
        ":cost_analyzer_lib",
        ":model_analyzer_lib",
        ":cpp_python_util",
        ":cpp_shape_inference",
        ":kernel_registry",
        ":numpy_lib",
        ":safe_ptr",
        ":py_exception_registry",
        ":py_func_lib",
        ":py_record_reader_lib",
        ":py_record_writer_lib",
        ":python_op_gen",
        ":tf_session_helper",
        "//third_party/python_runtime:headers",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:python_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core/distributed_runtime/rpc:grpc_rpc_factory_registration",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_memory",
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:reader_base",
        "//tensorflow/core/debug",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/profiler/internal:print_model_analysis",
        "//tensorflow/tools/graph_transforms:transform_graph_lib",
        "//tensorflow/python/eager:pywrap_tfe_lib",
    ] + (tf_additional_lib_deps() +
         tf_additional_plugin_deps() +
         tf_additional_verbs_deps() +
         tf_additional_mpi_deps() +
         tf_additional_gdr_deps()) + if_ngraph([
        "@ngraph_tf//:ngraph_tf",
    ]),
)

# ** Targets for Windows build (start) **
# We need the following targets to expose symbols from _pywrap_tensorflow.dll

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
genrule(
    name = "pywrap_tensorflow_filtered_def_file",
    srcs = ["//tensorflow:tensorflow_def_file"],
    outs = ["pywrap_tensorflow_filtered_def_file.def"],
    cmd = select({
        "//tensorflow:windows": """
              $(location @local_config_def_file_filter//:def_file_filter) \\
              --input $(location //tensorflow:tensorflow_def_file) \\
              --output $@ \\
              --target _pywrap_tensorflow_internal.pyd
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    tools = ["@local_config_def_file_filter//:def_file_filter"],
    visibility = ["//visibility:public"],
)

# Get the import library of _pywrap_tensorflow_internal.pyd
filegroup(
    name = "get_pywrap_tensorflow_import_lib_file",
    srcs = [":_pywrap_tensorflow_internal.so"],
    output_group = "interface_library",
)

# Rename the import library for _pywrap_tensorflow_internal.pyd to _pywrap_tensorflow_internal.lib
# (It was _pywrap_tensorflow_internal.so.if.lib).
genrule(
    name = "pywrap_tensorflow_import_lib_file",
    srcs = [":get_pywrap_tensorflow_import_lib_file"],
    outs = ["_pywrap_tensorflow_internal.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# Create a cc_import rule for the import library of _pywrap_tensorflow_internal.dll
# so that custom ops' dynamic libraries can link against it.
cc_import(
    name = "pywrap_tensorflow_import_lib",
    interface_library = select({
        "//tensorflow:windows": ":pywrap_tensorflow_import_lib_file",
        "//conditions:default": "not_exsiting_on_unix.lib",  # Just a placeholder for Unix platforms
    }),
    system_provided = 1,
)

# ** Targets for Windows build (end) **

py_library(
    name = "lib",
    srcs = [
        "lib/io/file_io.py",
        "lib/io/python_io.py",
        "lib/io/tf_record.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":pywrap_tensorflow",
        ":util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "session",
    srcs = ["client/session.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":c_api_util",
        ":error_interpolation",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
        ":pywrap_tensorflow",
        ":session_ops",
        ":util",
        "//tensorflow/python:mixed_precision_global_state",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "server_lib_test",
    size = "small",
    srcs = ["training/server_lib_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
)

tf_py_test(
    name = "server_lib_multiple_containers_test",
    size = "small",
    srcs = ["training/server_lib_multiple_containers_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
)

tf_py_test(
    name = "server_lib_same_variables_clear_container_test",
    size = "small",
    srcs = ["training/server_lib_same_variables_clear_container_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
)

tf_py_test(
    name = "server_lib_same_variables_clear_test",
    size = "small",
    srcs = ["training/server_lib_same_variables_clear_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
)

tf_py_test(
    name = "server_lib_same_variables_no_clear_test",
    size = "small",
    srcs = ["training/server_lib_same_variables_no_clear_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
)

tf_py_test(
    name = "server_lib_sparse_job_test",
    size = "small",
    srcs = ["training/server_lib_sparse_job_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
)

cuda_py_test(
    name = "localhost_cluster_performance_test",
    size = "medium",
    srcs = [
        "training/localhost_cluster_performance_test.py",
    ],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":distributed_framework_test_lib",
        ":framework_for_generated_wrappers",
        ":partitioned_variables",
        ":training",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
    ],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "oss_serial",
    ],
    xla_enable_strict_auto_jit = True,
)

tf_py_test(
    name = "sync_replicas_optimizer_test",
    size = "medium",
    srcs = [
        "training/sync_replicas_optimizer_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":training",
        ":variables",
    ],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "notsan",  # data race due to b/62910646
        "oss_serial",
    ],
)

py_library(
    name = "timeline",
    srcs = ["client/timeline.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
    ],
)

# Just used by tests.
tf_cuda_library(
    name = "construction_fails_op",
    srcs = ["client/test_construction_fails_op.cc"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_py_test(
    name = "session_test",
    size = "medium",
    srcs = ["client/session_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":config",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":training",
        ":util",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
    grpc_enabled = True,
    tags = [
        "no_gpu",  # b/127001953
        "no_pip_gpu",  # testInteractivePlacePrunedGraph fails on invalid assumption about GPU ops.
        "no_windows",
    ],
)

tf_py_test(
    name = "session_clusterspec_prop_test",
    size = "small",
    srcs = ["client/session_clusterspec_prop_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":training",
        ":util",
        ":variables",
        "//third_party/py/numpy",
    ],
    grpc_enabled = True,
    tags = [
        "no_gpu",
        "no_oss",
        "no_pip",
        "no_pip_gpu",
        "notap",
    ],
)

tf_py_test(
    name = "session_list_devices_test",
    size = "small",
    srcs = ["client/session_list_devices_test.py"],
    additional_deps = [
        ":client",
        ":framework",
        ":framework_test_lib",
        ":platform_test",
        ":training",
    ],
    grpc_enabled = True,
    tags = [
        "no_gpu",
        "no_pip_gpu",
        "notsan",  # data race due to b/62910646
    ],
)

tf_py_test(
    name = "session_partial_run_test",
    size = "small",
    srcs = ["client/session_partial_run_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":training",
        ":util",
        "@six_archive//:six",
    ],
    grpc_enabled = True,
    tags = [
        "no_gpu",
        "no_windows",
    ],
)

cuda_py_test(
    name = "timeline_test",
    size = "small",
    srcs = ["client/timeline_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
    ],
    xla_enable_strict_auto_jit = False,  # Graph structure is different with autojit
)

cuda_py_test(
    name = "virtual_gpu_test",
    size = "small",
    srcs = ["client/virtual_gpu_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "no_gpu",  # b/127386241
        "no_windows_gpu",
    ],
    xla_enable_strict_auto_jit = True,
)

tf_py_test(
    name = "c_api_util_test",
    size = "small",
    srcs = ["framework/c_api_util_test.py"],
    additional_deps = [
        ":c_api_util",
        ":framework_test_lib",
        ":platform_test",
    ],
)

tf_py_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["framework/graph_util_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":state_ops_gen",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "convert_to_constants_test",
    size = "small",
    srcs = ["framework/convert_to_constants_test.py"],
    additional_deps = [
        ":convert_to_constants",
        "client_testlib",
        "framework_test_lib",
    ],
)

tf_py_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["lib/core/bfloat16_test.py"],
    additional_deps = [
        ":client_testlib",
        ":lib",
        ":pywrap_tensorflow",
    ],
)

tf_py_test(
    name = "file_io_test",
    size = "small",
    srcs = ["lib/io/file_io_test.py"],
    additional_deps = [
        ":client_testlib",
        ":errors",
        ":lib",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "tf_record_test",
    size = "small",
    srcs = ["lib/io/tf_record_test.py"],
    additional_deps = [
        ":client_testlib",
        ":errors",
        ":lib",
        ":util",
    ],
)

cuda_py_test(
    name = "adam_test",
    size = "small",
    srcs = ["training/adam_test.py"],
    additional_deps = [
        ":array_ops",
        ":framework",
        ":math_ops",
        ":platform",
        ":training",
        ":platform_test",
        ":client_testlib",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "moving_averages_test",
    size = "small",
    srcs = [
        "training/moving_averages_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":constant_op",
        ":dtypes",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":training",
        ":variable_scope",
        ":variables",
    ],
    tags = ["notsan"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_tests(
    name = "training_tests",
    size = "medium",
    srcs = [
        "training/adadelta_test.py",
        "training/adagrad_da_test.py",
        "training/adagrad_test.py",
        "training/basic_loops_test.py",
        "training/coordinator_test.py",
        "training/device_setter_test.py",
        "training/ftrl_test.py",
        "training/gradient_descent_test.py",
        "training/learning_rate_decay_test.py",
        "training/momentum_test.py",
        "training/optimizer_test.py",
        "training/proximal_adagrad_test.py",
        "training/proximal_gradient_descent_test.py",
        "training/quantize_training_test.py",
        "training/queue_runner_test.py",
        "training/rmsprop_test.py",
        "training/slot_creator_test.py",
        "training/tensorboard_logging_test.py",
        "training/training_ops_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":data_flow_ops_gen",
        ":embedding_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":lookup_ops",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":partitioned_variables",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":resources",
        ":sparse_ops",
        ":state_ops",
        ":state_ops_gen",
        ":summary",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "saver_test_utils",
    srcs = ["training/saver_test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lookup_ops_gen",
        ":training",
    ],
)

cuda_py_test(
    name = "saver_test",
    size = "medium",
    srcs = [
        "training/saver_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":saver_test_utils",
        ":partitioned_variables",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":sparse_ops",
        ":summary",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = ["multi_gpu"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "checkpoint_management_test",
    size = "small",
    srcs = [
        "training/checkpoint_management_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":saver_test_utils",
        ":partitioned_variables",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":sparse_ops",
        ":summary",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    xla_enable_strict_auto_jit = True,
)

tf_py_test(
    name = "saver_large_variable_test",
    size = "medium",
    srcs = ["training/saver_large_variable_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":errors",
        ":framework_for_generated_wrappers",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "manual",
        "noasan",  # http://b/30379628
        "notsan",  # http://b/30379628
    ],
)

tf_py_test(
    name = "saver_large_partitioned_variable_test",
    size = "medium",
    srcs = ["training/saver_large_partitioned_variable_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":partitioned_variables",
        ":training",
        ":variables",
    ],
    tags = [
        "noasan",  # http://b/30782289
        "notsan",  # http://b/30782289
    ],
)

cuda_py_test(
    name = "session_manager_test",
    size = "medium",  # TODO(irving): Can this be made small?
    srcs = ["training/session_manager_test.py"],
    additional_deps = [
        ":array_ops",
        ":control_flow_ops",
        ":client",
        ":client_testlib",
        ":errors",
        ":framework_for_generated_wrappers",
        ":platform",
        ":training",
        ":variables",
    ],
    grpc_enabled = True,
    main = "training/session_manager_test.py",
    xla_enable_strict_auto_jit = True,
)

tf_py_test(
    name = "supervisor_test",
    size = "small",
    srcs = ["training/supervisor_test.py"],
    additional_deps = [
        ":array_ops",
        ":checkpoint_management",
        ":client_testlib",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":parsing_ops",
        ":platform",
        ":saver",
        ":summary",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
    grpc_enabled = True,
    tags = ["no_windows"],
)

tf_py_test(
    name = "basic_session_run_hooks_test",
    size = "medium",
    srcs = ["training/basic_session_run_hooks_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":nn_grad",
        ":platform",
        ":state_ops",
        ":summary",
        ":training",
        ":variable_scope",
        ":variables",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "no_pip",  # Relies on contrib
        "no_windows",
        "notsan",  # intermittent races on a few percent of runs
    ],
)

tf_py_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["training/checkpoint_utils_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":partitioned_variables",
        ":platform",
        ":pywrap_tensorflow",
        ":resource_variable_ops",
        ":state_ops",
        ":training",
        ":variable_scope",
        ":variables",
    ],
    tags = [
        "manual",
        "no_cuda_on_cpu_tap",
        "no_oss",
        "no_windows",
        "notap",
    ],
)

tf_py_test(
    name = "checkpoint_ops_test",
    size = "small",
    srcs = ["training/checkpoint_ops_test.py"],
    additional_deps = [
        ":checkpoint_ops_gen",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":partitioned_variables",
        ":platform",
        ":pywrap_tensorflow",
        ":state_ops",
        ":training",
        ":variable_scope",
        ":variables",
    ],
)

tf_py_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["training/warm_starting_util_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":dtypes",
        ":framework_ops",
        ":init_ops",
        ":training",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "monitored_session_test",
    size = "medium",
    srcs = ["training/monitored_session_test.py"],
    additional_deps = [
        ":array_ops",
        ":checkpoint_management",
        ":client_testlib",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":resource_variable_ops",
        ":saver",
        ":session",
        ":state_ops",
        ":summary",
        ":training",
        ":variables",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_coordinator",
    ],
    tags = [
        "no_pip",
        "notsan",  # b/67945581
    ],
)

py_library(
    name = "training_util",
    srcs = ["training/training_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework",
        ":framework_ops",
        ":init_ops",
        ":platform",
        ":resource_variable_ops",
        ":state_ops",
        ":util",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/eager:context",
    ],
)

tf_py_test(
    name = "training_util_test",
    size = "small",
    srcs = ["training/training_util_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework",
        ":platform",
        ":training",
        ":training_util",
        ":variables",
    ],
)

tf_py_test(
    name = "input_test",
    size = "medium",
    srcs = ["training/input_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        ":util",
        ":variables",
        ":training",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "summary_op_util",
    srcs = ["ops/summary_op_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
    ],
)

py_library(
    name = "summary",
    srcs = glob(
        ["summary/**/*.py"],
        exclude = ["**/*test*"],
    ),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":constant_op",
        ":framework",
        ":framework_for_generated_wrappers",
        ":lib",
        ":logging_ops_gen",
        ":platform",
        ":protos_all_py",
        ":pywrap_tensorflow",
        ":summary_op_util",
        ":summary_ops_gen",
        ":summary_ops_v2",
        ":util",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_tests(
    name = "summary_tests",
    size = "small",
    srcs = [
        "summary/plugin_asset_test.py",
        "summary/summary_test.py",
        "summary/writer/writer_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":variables",
        ":framework",
        ":framework_test_lib",
        ":platform",
        ":platform_test",
        ":summary",
        ":summary_ops_v2",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "layers_base",
    srcs = [
        "layers/__init__.py",
        "layers/base.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":layers_util",
        ":platform",
        ":smart_cond",
        ":tensor_util",
        ":util",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:engine",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "layers_util",
    srcs = [
        "layers/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":smart_cond",
        ":util",
        ":variables",
    ],
)

py_library(
    name = "layers",
    srcs = [
        "layers/convolutional.py",
        "layers/core.py",
        "layers/layers.py",
        "layers/normalization.py",
        "layers/pooling.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":init_ops",
        ":layers_base",
        ":math_ops",
        ":nn",
        ":nn_ops",
        ":platform",
        ":resource_variable_ops",
        ":resource_variable_ops_gen",
        ":standard_ops",
        ":state_ops",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "layers_base_test",
    size = "small",
    srcs = ["layers/base_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":init_ops",
        ":layers",
        ":layers_base",
        ":math_ops",
        ":random_ops",
        ":variable_scope",
        "//tensorflow/python/eager:context",
    ],
    main = "layers/base_test.py",
)

tf_py_test(
    name = "layers_core_test",
    size = "small",
    srcs = ["layers/core_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":layers",
        ":math_ops",
        ":nn_ops",
        ":random_ops",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
    ],
    main = "layers/core_test.py",
)

tf_py_test(
    name = "layers_convolutional_test",
    size = "small",
    srcs = ["layers/convolutional_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":layers",
        ":math_ops",
        ":nn_ops",
        ":random_ops",
    ],
    main = "layers/convolutional_test.py",
)

tf_py_test(
    name = "layers_utils_test",
    size = "small",
    srcs = ["layers/utils_test.py"],
    additional_deps = [
        ":client_testlib",
        ":layers",
    ],
    main = "layers/utils_test.py",
)

tf_py_test(
    name = "layers_pooling_test",
    size = "small",
    srcs = ["layers/pooling_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_test_lib",
        ":layers",
        ":random_ops",
    ],
    main = "layers/pooling_test.py",
)

cuda_py_test(
    name = "layers_normalization_test",
    size = "medium",
    srcs = ["layers/normalization_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":layers",
        ":math_ops",
        ":random_ops",
        ":variables",
        "//third_party/py/numpy",
    ],
    main = "layers/normalization_test.py",
    shard_count = 10,
    xla_enable_strict_auto_jit = True,
)

# -----------------------------------------------------------------------------
# Quantization

tf_py_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["ops/dequantize_op_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "quantized_ops_test",
    size = "small",
    srcs = ["ops/quantized_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["ops/quantized_conv_ops_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
    tags = ["no_windows"],
)

cuda_py_test(
    name = "accumulate_n_benchmark",
    size = "medium",
    srcs = ["ops/accumulate_n_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops_gen",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":random_ops",
        ":state_ops",
        ":state_ops_gen",
    ],
    main = "ops/accumulate_n_benchmark.py",
    shard_count = 6,
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "batch_norm_benchmark",
    srcs = ["ops/batch_norm_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        ":platform",
        ":random_ops",
        ":variables",
    ],
    main = "ops/batch_norm_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "concat_benchmark",
    srcs = ["ops/concat_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/concat_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "control_flow_ops_benchmark",
    srcs = ["ops/control_flow_ops_benchmark.py"],
    additional_deps = [
        ":client_testlib",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        "//tensorflow/python/eager:function",
    ],
    main = "ops/control_flow_ops_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "conv2d_benchmark",
    size = "large",
    srcs = ["ops/conv2d_benchmark.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":nn_ops",
        ":platform",
        ":platform_benchmark",
        ":random_ops",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/conv2d_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "split_benchmark",
    srcs = ["ops/split_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/split_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "transpose_benchmark",
    size = "medium",
    srcs = ["ops/transpose_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/transpose_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "matmul_benchmark",
    size = "medium",
    srcs = ["ops/matmul_benchmark.py"],
    additional_deps = [":matmul_benchmark_main_lib"],
    main = "ops/matmul_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "matmul_benchmark_main_lib",
    testonly = True,
    srcs = ["ops/matmul_benchmark.py"],
    deps = [
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform",
        ":platform_benchmark",
        ":random_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "matmul_benchmark_test",
    size = "medium",
    srcs = ["ops/matmul_benchmark_test.py"],
    additional_deps = [
        ":math_ops",
        ":random_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":matmul_benchmark",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/matmul_benchmark_test.py",
    tags = ["no_pip"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "session_benchmark",
    srcs = ["client/session_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":random_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
    ],
    grpc_enabled = True,
    main = "client/session_benchmark.py",
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "nn_grad_test",
    size = "small",
    srcs = ["ops/nn_grad_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn_grad",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "tf_item",
    srcs = [
        "grappler/item.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_tensorflow_internal",
        "//tensorflow/core/grappler/costs:op_performance_data_py",
    ],
)

tf_py_test(
    name = "item_test",
    size = "small",
    srcs = [
        "grappler/item_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
)

tf_py_test(
    name = "datasets_test",
    size = "small",
    srcs = [
        "grappler/datasets_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data",
    ],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
)

py_library(
    name = "tf_cluster",
    srcs = [
        "grappler/cluster.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_tensorflow_internal",
        "//tensorflow/core/grappler/costs:op_performance_data_py",
    ],
)

cuda_py_test(
    name = "cluster_test",
    size = "small",
    srcs = [
        "grappler/cluster_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":tf_cluster",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
    ],
    shard_count = 10,
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
        "notap",  # TODO(b/135924227): Re-enable after fixing flakiness.
    ],
    # This test will not run on XLA because it primarily tests the TF Classic flow.
    xla_enable_strict_auto_jit = False,
)

py_library(
    name = "tf_optimizer",
    srcs = [
        "grappler/tf_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_tensorflow_internal",
        ":tf_cluster",
    ],
)

tf_py_test(
    name = "tf_optimizer_test",
    size = "small",
    srcs = [
        "grappler/tf_optimizer_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":tf_item",
        ":tf_optimizer",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
)

py_library(
    name = "graph_placer",
    srcs = [
        "grappler/controller.py",
        "grappler/graph_placer.py",
        "grappler/hierarchical_controller.py",
    ],
    deps = [
        ":python",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "graph_placer_test",
    size = "large",
    srcs = ["grappler/graph_placer_test.py"],
    additional_deps = [
        ":client_testlib",
        ":graph_placer",
        "//tensorflow/python:math_ops",
    ],
    tags = [
        "grappler",
        "no_pip",  # graph_placer is not available in pip.
    ],
)

tf_py_test(
    name = "memory_optimizer_test",
    size = "medium",
    srcs = [
        "grappler/memory_optimizer_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":random_seed",
        ":session",
        ":tf_optimizer",
        ":training",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
    ],
)

cuda_py_test(
    name = "constant_folding_test",
    size = "medium",
    srcs = [
        "grappler/constant_folding_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":functional_ops",
        ":math_ops",
        ":ops",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "layout_optimizer_test",
    size = "medium",
    srcs = [
        "grappler/layout_optimizer_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":functional_ops",
        ":math_ops",
        ":nn",
        ":ops",
        ":random_ops",
        ":state_ops",
        ":tf_cluster",
        ":tf_optimizer",
        ":training",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    shard_count = 10,
    tags = [
        "grappler",
    ],
    # This test will not run on XLA because it primarily tests the TF Classic flow.
    xla_enable_strict_auto_jit = False,
)

py_library(
    name = "cost_analyzer",
    srcs = [
        "grappler/cost_analyzer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow_internal",
        ":tf_cluster",
        ":tf_item",
    ],
)

py_binary(
    name = "cost_analyzer_tool",
    srcs = [
        "grappler/cost_analyzer_tool.py",
    ],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":cost_analyzer",
        ":framework_for_generated_wrappers",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "cost_analyzer_test",
    size = "small",
    srcs = ["grappler/cost_analyzer_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":cost_analyzer",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":random_ops",
        ":state_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
        "no_cuda_on_cpu_tap",
        "no_pip",
        "nomac",
    ],
)

py_library(
    name = "model_analyzer",
    srcs = [
        "grappler/model_analyzer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":pywrap_tensorflow_internal"],
)

tf_py_test(
    name = "model_analyzer_test",
    size = "small",
    srcs = ["grappler/model_analyzer_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":model_analyzer",
        ":state_ops",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
        "no_pip",
    ],
)

cuda_py_test(
    name = "auto_mixed_precision_test",
    size = "small",
    srcs = [
        "grappler/auto_mixed_precision_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":math_ops",
        ":nn",
        ":ops",
        ":random_ops",
        ":control_flow_ops",
        ":training",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    tags = [
        "grappler",
        "no_rocm",
    ],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
)

tf_gen_op_wrapper_private_py(
    name = "nccl_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:nccl_ops_op_lib",
    ],
)

py_library(
    name = "nccl_ops",
    srcs = ["ops/nccl_ops.py"],
    srcs_version = "PY2AND3",
    visibility = visibility + [
        "//learning/deepmind/tensorflow:__subpackages__",
    ],
    deps = [
        ":framework_for_generated_wrappers",
        ":nccl_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
    ],
)

cuda_py_test(
    name = "nccl_ops_test",
    size = "small",
    srcs = ["ops/nccl_ops_test.py"],
    additional_deps = [
        ":nccl_ops",
        ":array_ops",
        ":client_testlib",
        ":framework_test_lib",
        ":platform_test",
    ],
    # Disabled on jenkins until errors finding nvmlShutdown are found.
    tags = [
        "manual",
        "multi_gpu",
        "no_oss",
        "noguitar",
        "notap",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "decode_proto_ops_gen",
    deps = [
        "//tensorflow/core:decode_proto_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "encode_proto_ops_gen",
    deps = [
        "//tensorflow/core:encode_proto_ops_op_lib",
    ],
)

py_library(
    name = "proto_ops",
    srcs = ["ops/proto_ops.py"],
    deps = [
        ":decode_proto_ops_gen",
        ":encode_proto_ops_gen",
        "//tensorflow/python:framework_ops",
    ],
)

py_binary(
    name = "graph_analyzer",
    srcs = [
        "grappler/graph_analyzer.py",
    ],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":pywrap_tensorflow_internal",
    ],
)

pyx_library(
    name = "framework_fast_tensor_util",
    srcs = ["framework/fast_tensor_util.pyx"],
    py_deps = ["//tensorflow/python:util"],
    deps = ["//third_party/py/numpy:headers"],
)

py_library(
    name = "tf2",
    srcs = ["tf2.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tf2_test",
    srcs = ["framework/tf2_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":tf2",
        "//tensorflow/python/distribute:combinations",
    ],
)

cuda_py_test(
    name = "raw_ops_test",
    srcs = ["ops/raw_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
    ],
    xla_enable_strict_auto_jit = True,
)
