BINDIR := bin
SRCDIR := src
LOGDIR := log
LIBDIR := lib
TESTDIR := test


# Source code file extension
SRCEXT := c


# Defines the C Compiler
CC := gcc


# Defines the language standards for GCC
STD := -std=gnu99 # See man gcc for more options

# Protection for stack-smashing attack
STACK := -fstack-protector-all -Wstack-protector

# Specifies to GCC the required warnings
WARNS := -Wall -Wextra -pedantic # -pedantic warns on language standards

# Flags for compiling
#CFLAGS := -O3 $(STD) $(STACK) $(WARNS)

# Debug options
#DEBUG := -g3 -DDEBUG=1

# Dependency libraries
LIBS := -lm  -lpthread#-I some/path/to/library









# %.o file names
NAMES := $(notdir $(basename $(wildcard $(SRCDIR)/*.$(SRCEXT))))
OBJECTS :=

#
# COMPILATION RULES
#

default: all


# Rule for link and generate the binary file
all: 
	@echo -en "$(BROWN)LD $(END_COLOR)";
	$(CC) -o $(BINDIR)/$(BINARY) $+ $(DEBUG) $(CFLAGS) $(LIBS)
	@echo -en "\n--\nBinary file placed at" \
			  "$(BROWN)$(BINDIR)/$(BINARY)$(END_COLOR)\n";


# Rule for object binaries compilation
$(LIBDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo -en "$(BROWN)CC $(END_COLOR)";
	$(CC)  $^ -o $@ $(DEBUG) $(CFLAGS) $(LIBS)


# Rule for run valgrind tool
run:
	gcc -o lib/nn.o src/neuralnet.c -lm
	lib/./nn.o $(arg)
	gnuplot src/plot.gnu



# Compile tests and run the test binary
# @rm -rvf $(BINDIR)/* $(LIBDIR)/* $(LOGDIR)/*;

# Rule for cleaning the project
clean:
	rm lib/nn.o
