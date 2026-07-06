class FindDuplicates < Formula
  include Language::Python::Virtualenv

  desc "Find and review duplicate images by perceptual hash and quality metrics"
  homepage "https://github.com/pathanin/find_duplicates"
  url "https://github.com/pathanin/find_duplicates/releases/download/v0.1.0/find_duplicates-0.1.0.tar.gz"
  sha256 "65b1e9a34d9aff17ffb0a150347cbafd172499acc4e10760ad4f32a0d0b3a2be"
  license "MIT"

  depends_on "python@3.13"

  # Perceptual hashing and image I/O
  resource "numpy" do
    url "https://files.pythonhosted.org/packages/50/8e/b8041bc719f056afd864478029d52214789341ac6583437b0ee5031e9530/numpy-2.4.5.tar.gz"
    sha256 "ca670567a5683b7c1670ec03e0ddd5862e10934e92a70751d68d7b7b74ca7f9f"
  end

  resource "opencv-python-headless" do
    url "https://files.pythonhosted.org/packages/1d/99/76b7c80252aa83c1af16393454aafd125a0287101afe8deb0a6821af0e30/opencv_python_headless-5.0.0.93.tar.gz"
    sha256 "b82f9831daab90b725c7c1ee1b36cb5732c367096ac76d119e64e14eb70d5f3c"
  end

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/8c/21/c2bcdd5906101a30244eaffc1b6e6ce71a31bd0742a01eb89e660ebfac2d/pillow-12.2.0.tar.gz"
    sha256 "a830b1a40919539d07806aa58e1b114df53ddd43213d9c8b75847eee6c0182b5"
  end

  # Terminal UI framework
  resource "textual" do
    url "https://files.pythonhosted.org/packages/19/89/bec5709fb759f9c784bbcb30b2e3497df3f901691d13c2b864dbf6694a17/textual-8.2.4.tar.gz"
    sha256 "d4e2b2ddd7157191d00b228592b7c739ea080b7d792fd410f23ca75f05ea76c4"
  end

  resource "textual-image" do
    url "https://files.pythonhosted.org/packages/10/77/b2128ced69556bfbb8e1c19d8f013e621cf12531eaba4e9b09e1cfa81e37/textual_image-0.13.2.tar.gz"
    sha256 "8ca0cee2bfcd7734de5b16a1936da226b77b745e28830d9cf2bc202cb70e43ee"
  end

  # Textual transitive dependencies
  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl"
    sha256 "355216845c60bd96232cd8d8c40e8f9765cc86f46880e43a8fd22dc1a1a8cab1"
  end

  resource "mdit-py-plugins" do
    url "https://files.pythonhosted.org/packages/fb/86/dd6e5db36df29e76c7a7699123569a4a18c1623ce68d826ed96c62643cae/mdit_py_plugins-0.5.0-py3-none-any.whl"
    sha256 "07a08422fc1936a5d26d146759e9155ea466e842f5ab2f7d2266dd084c8dab1f"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl"
    sha256 "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8"
  end

  resource "platformdirs" do
    url "https://files.pythonhosted.org/packages/3c/a6/bc1012356d8ece4d66dd75c4b9fc6c1f6650ddd5991e421177d9f8f671be/platformdirs-4.3.6-py3-none-any.whl"
    sha256 "73e575e1408ab8103900836b97580d5307456908a03e92031bab39e4554cc3fb"
  end

  resource "pygments" do
    url "https://files.pythonhosted.org/packages/f4/7e/a72dd26f3b0f4f2bf1dd8923c85f7ceb43172af56d63c7383eb62b332364/pygments-2.20.0-py3-none-any.whl"
    sha256 "81a9e26dd42fd28a23a2d169d86d7ac03b46e2f8b59ed4698fb4785f946d0176"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/19/71/39c7c0d87f8d4e6c020a393182060eaefeeae6c01dab6a84ec346f2567df/rich-13.9.4-py3-none-any.whl"
    sha256 "6049d5e6ec054bf2779ab3358186963bac2ea89175919d699e378b99738c2a90"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl"
    sha256 "f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548"
  end

  def install
    virtualenv_install_with_resources
    # Install the scripts into the virtualenv
    libexec.install "find_duplicates.py", "compare_image_quality.py"
    # Create a wrapper so 'find-duplicates' works from PATH
    (bin/"find-duplicates").write <<~EOS
      #!/bin/bash
      exec "#{libexec}/bin/python3" "#{libexec}/find_duplicates.py" "$@"
    EOS
  end

  test do
    system bin/"find-duplicates", "--help"
  end
end
