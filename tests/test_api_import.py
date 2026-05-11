def test_api_import():
    from rental_price_mlops.api.main import app
    assert app is not None