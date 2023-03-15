B_TEMP = `$(XF_PROJ_ROOT)/common/utility/parse_platform_list.py $(PLATFORM)`

#Setting Platform Path
ifeq ($(findstring xpfm, $(PLATFORM)), xpfm)
	B_NAME = $(shell dirname $(PLATFORM))
else
	B_NAME = $(B_TEMP)/$(PLATFORM)
endif

VIVADO := $(XILINX_VIVADO)/bin/vivado
$(TEMP_DIR)/myadder.xo: 
	mkdir -p $(TEMP_DIR)
	$(VIVADO) -mode batch -source ./src/gen_xo.tcl -tclargs $(TEMP_DIR)/myadder.xo myadder $(TARGET) $(B_NAME)/$(XSA).xpfm $(XSA)
$(TEMP_DIR)/krnl_s2mm.xo: ./src/krnl_s2mm.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(VPP_FLAGS) -c --platform $(PLATFORM) -k krnl_s2mm -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/krnl_mm2s.xo: ./src/krnl_mm2s.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(VPP_FLAGS) -c --platform $(PLATFORM) -k krnl_mm2s -I'$(<D)' -o'$@' '$<'
