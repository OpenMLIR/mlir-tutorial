set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name Toy)
add_public_tablegen_target(MLIRToyPassIncGen)

add_mlir_doc(Passes ToyPasses ./ -gen-pass-doc)
