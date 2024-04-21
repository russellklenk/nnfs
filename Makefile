DEBUG_FLAGS              = -DDEBUG -g
OPTIMIZER_FLAGS          = -O0
CC                       = gcc
CXX                      = g++
AR                       = ar
STRIP                    = strip

## Targets
TARGET_FORWARD           = forward

WARNINGS                 = -Wall -Wextra -Werror=return-type
INCLUDE_DIRS             = -I. -Iinclude
CFLAGS_BASE              = -std=c99 -pthread -fstrict-aliasing -D__STDC_FORMAT_MACROS
CXXFLAGS_BASE            = -std=c++11 -pthread -fstrict-aliasing -D__STDC_FORMAT_MACROS

SOURCE_FILES_FORWARD     = forward.c
OBJECT_FILES_FORWARD     = $(SOURCE_FILES_FORWARD:.c=.o)
DEPENDENCY_FILES_FORWARD = $(OBJECT_FILES_FORWARD:.o=.d)

LOG_FILES                = $(wildcard *.log) $(wildcard *.log.bak)

ifeq ($(strip $(STRIP_SYMBOLS)), 1)
	STRIP_SYMBOLS_CMD    = $(STRIP) --strip-all $(TARGET)
else
	STRIP_SYMBOLS_CMD    = 
endif

$(TARGET_FORWARD): $(OBJECT_FILES_FORWARD)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_BASE) $(FLOAT_FLAGS) $(DEBUG_FLAGS) $(RELEASE_FLAGS) $(TEST_FLAGS) $(INCLUDE_DIRS) $(WARNINGS) $(OPTIMIZER_FLAGS) -o $@ $^ $(LDFLAGS) -lrt

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_BASE) $(FLOAT_FLAGS) $(DEBUG_FLAGS) $(RELEASE_FLAGS) $(TEST_FLAGS) $(INCLUDE_DIRS) $(WARNINGS) $(OPTIMIZER_FLAGS) -o $@ -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_BASE) $(FLOAT_FLAGS) $(DEBUG_FLAGS) $(RELEASE_FLAGS) $(TEST_FLAGS) $(INCLUDE_DIRS) $(WARNINGS) $(OPTIMIZER_FLAGS) -o $@ -c $<

-include $(DEPENDENCY_FILES)

%.d: %.c
	@$(CPP) $(CFLAGS) $(CFLAGS_BASE) $(FLOAT_FLAGS) $(DEBUG_FLAGS) $(RELEASE_FLAGS) $(TEST_FLAGS) $(INCLUDE_DIRS) $< -MM -MT $(@:.d=.o) >$@

%.d: %.cpp
	@$(CPP) $(CXXFLAGS) $(CXXFLAGS_BASE) $(FLOAT_FLAGS) $(DEBUG_FLAGS) $(RELEASE_FLAGS) $(TEST_FLAGS) $(INCLUDE_DIRS) $< -MM -MT $(@:.d=.o) >$@

.PHONY: all
all: $(TARGET_FORWARD)

.PHONY: clean
clean:
	rm -f $(LOG_FILES) $(OBJECT_FILES_FORWARD) $(DEPENDENCY_FILES_FORWARD) $(TARGET_FORWARD)

.PHONY: cleandeps
cleandeps:
	rm -f $(DEPENDENCY_FILES_FORWARD)

