.PHONY: serve optimize

serve:
	$(MAKE) -C blog serve

optimize:
ifndef IMAGES_PATH
	$(error IMAGES_PATH is required. Usage: make optimize IMAGES_PATH=blog/content/colorimetry/laser-power-meter)
endif
	exiftool -gps:all= -overwrite_original -P "$(IMAGES_PATH)/"*.jpg
	magick mogrify -resize "1024x1024>" -quality 85 "$(IMAGES_PATH)/"*.jpg
