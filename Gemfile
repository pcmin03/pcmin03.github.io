# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll", "~> 4.3"
gem "jekyll-seo-tag"
gem "jekyll-sitemap"
gem "jekyll-feed"

group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-archives"
end

gem "html-proofer", "~> 5.0", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
