

export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"

export OpenMP_ROOT=$(brew --prefix)/opt/libomp
export LDFLAGS="${LDFLAGS} -L${OpenMP_ROOT}/lib"
export CPPFLAGS="${CPPFLAGS} -I${OpenMP_ROOT}/include"

export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
#export CPPFLAGS="-I/opt/homebrew/include /opt/homebrew/opt/llvm/include /opt/homebrew/opt/libomp/include"
#export LDFLAGS="-L/opt/homebrew/lib -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib"


# export CMAKE_PREFIX_PATH="/opt/homebrew/opt/llvm"


# cmake .. -DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++


#export CC=/opt/homebrew/opt/llvm/bin/clang
#export CXX=/opt/homebrew/opt/llvm/bin/clang++
#export CPPFLAGS="-I/opt/homebrew/include"
#export LDFLAGS="-L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib"
#cmake ..


