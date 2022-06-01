test: clean
	python3 process_midi.py
PHONY:
clean:
	-@rm outputs/*
